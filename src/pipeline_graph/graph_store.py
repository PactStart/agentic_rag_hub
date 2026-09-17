"""把三元组写入 Neo4j（MERGE，避免重复节点）。"""

from __future__ import annotations

import os

from neo4j import GraphDatabase


class GraphStore:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def write(self, triples: list[dict]) -> int:
        cypher = (
            "MERGE (a:Entity {name: $h}) "
            "MERGE (b:Entity {name: $t}) "
            "MERGE (a)-[r:REL {type: $rel}]->(b) "
            "SET r.source = $source, r.acl = $acl"
        )
        n = 0
        with self.driver.session() as session:
            for row in triples:
                session.run(
                    cypher,
                    h=row["h"],
                    t=row["t"],
                    rel=row["rel"],
                    source=row.get("source") or "",
                    acl=list(row.get("acl") or ["all"]),
                )
                n += 1
        return n

    def neighbors(self, names: list[str], hops: int = 2, limit: int = 20) -> list[dict]:
        if not names:
            return []
        hops = max(1, min(int(hops), 3))
        cypher = (
            "MATCH (e:Entity) WHERE e.name IN $names "
            f"MATCH path = (e)-[:REL*1..{hops}]-(n:Entity) "
            "UNWIND relationships(path) AS rel "
            "RETURN DISTINCT e.name AS seed, rel.type AS rel, "
            "startNode(rel).name AS h, endNode(rel).name AS t, "
            "rel.source AS source, rel.acl AS acl "
            "LIMIT $limit"
        )
        with self.driver.session() as session:
            result = session.run(cypher, names=names, limit=limit)
            rows = []
            seen: set[tuple] = set()
            for rec in result:
                key = (rec["h"], rec["rel"], rec["t"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "h": rec["h"],
                        "rel": rec["rel"],
                        "t": rec["t"],
                        "source": rec.get("source") or "",
                        "acl": list(rec.get("acl") or ["all"]),
                        "seed": rec["seed"],
                    }
                )
            return rows
