# TFID-data

This repository contains the data assets for the TFID (Transformer Forward Impact Dataset) research project.

## Overview

This repository hosts a Neo4j graph database dump containing structured data for TFID research purposes. The data is stored in a binary dump format that can be imported into a Neo4j database instance.

## Repository Structure

```
TFID-data/
├── tfid-data/
│   └── neo4j.dump          # Neo4j database dump file (~38 MB)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Contents

### Neo4j Database Dump

- **File**: `tfid-data/neo4j.dump`
- **Size**: ~38 MB
- **Format**: Neo4j binary dump format
- **Description**: Contains the complete graph database with nodes, relationships, and properties for the TFID dataset

## Prerequisites

To work with this data, you'll need:

- [Neo4j](https://neo4j.com/download/) (version 4.x or 5.x recommended)
- Neo4j Admin tools (included with Neo4j installation)

## Usage

### Importing the Database Dump

1. **Stop your Neo4j instance** (if running):
   ```bash
   neo4j stop
   ```

2. **Load the dump into a new database**:
   ```bash
   neo4j-admin database load --from-path=/path/to/TFID-data/tfid-data neo4j
   ```
   
   Or for Neo4j 4.x:
   ```bash
   neo4j-admin load --from=/path/to/TFID-data/tfid-data/neo4j.dump --database=neo4j --force
   ```

3. **Start Neo4j**:
   ```bash
   neo4j start
   ```

4. **Access the database**:
   - Open Neo4j Browser at `http://localhost:7474`
   - Or use Cypher queries via the Neo4j client

### Querying the Data

Once imported, you can query the database using Cypher. Example queries:

```cypher
// Count all nodes
MATCH (n) RETURN count(n)

// Count all relationships
MATCH ()-[r]->() RETURN count(r)

// View node labels
CALL db.labels()

// View relationship types
CALL db.relationshipTypes()

// Sample nodes
MATCH (n) RETURN n LIMIT 25
```

## Data Schema

To explore the data schema after importing:

```cypher
// View database schema
CALL db.schema.visualization()

// List all constraints
CALL db.constraints()

// List all indexes
CALL db.indexes()
```

## Development

### Ignored Files

The `.gitignore` file is configured to exclude:
- Local development dependencies (`data/`, `experiments/local/`)
- Build artifacts (`target/`, `build/`, `dist/`)
- IDE configurations (`.idea/`, `.vscode/`)
- Python cache files (`__pycache__/`, `*.pyc`)
- Environment files (`.env`, `venv/`)
- OS-specific files (`.DS_Store`)

## Contributing

If you need to update the database dump:

1. Make your changes in a Neo4j instance
2. Create a new dump:
   ```bash
   neo4j-admin database dump --to-path=/path/to/output neo4j
   ```
3. Replace the existing `neo4j.dump` file
4. Commit and push your changes

## License

[Add your license information here]

## Contact

For questions or issues related to this dataset, please open an issue in this repository.

## Repository

- **GitHub**: [akash1095/TFID-data](https://github.com/akash1095/TFID-data)

