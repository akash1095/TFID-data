# Neo4j Setup and Database Import Guide

This guide provides step-by-step instructions for installing Neo4j, loading the database dump, and configuring your environment for the TFID-data project.

## Table of Contents

- [Installation](#installation)
  - [macOS](#macos)
  - [Linux](#linux)
  - [Windows](#windows)
  - [Docker](#docker)
- [Loading the Database Dump](#loading-the-database-dump)
- [Starting Neo4j](#starting-neo4j)
- [Accessing Neo4j Browser](#accessing-neo4j-browser)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Installation

### macOS

#### Using Homebrew (Recommended)

```bash
# Install Neo4j
brew install neo4j

# Verify installation
neo4j version
```

#### Manual Installation

1. Download Neo4j Community Edition from [neo4j.com/download](https://neo4j.com/download/)
2. Extract the archive to your preferred location
3. Add Neo4j to your PATH:
   ```bash
   export PATH=$PATH:/path/to/neo4j/bin
   ```

### Linux

#### Using APT (Debian/Ubuntu)

```bash
# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list

# Update and install
sudo apt-get update
sudo apt-get install neo4j

# Verify installation
neo4j version
```

#### Using YUM (RedHat/CentOS)

```bash
# Add Neo4j repository
sudo rpm --import https://debian.neo4j.com/neotechnology.gpg.key
sudo cat <<EOF > /etc/yum.repos.d/neo4j.repo
[neo4j]
name=Neo4j RPM Repository
baseurl=https://yum.neo4j.com/stable
enabled=1
gpgcheck=1
EOF

# Install
sudo yum install neo4j

# Verify installation
neo4j version
```

### Windows

1. Download Neo4j Community Edition from [neo4j.com/download](https://neo4j.com/download/)
2. Run the installer and follow the installation wizard
3. Neo4j will be installed as a Windows service

### Docker

```bash
# Pull Neo4j image
docker pull neo4j:latest

# Run Neo4j container
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/your_password \
    neo4j:latest
```

## Loading the Database Dump

### Method 1: Using neo4j-admin (Recommended)

This method works for Neo4j 5.x:

```bash
# 1. Stop Neo4j if it's running
neo4j stop

# 2. Navigate to the repository directory
cd /path/to/TFID-data

# 3. Load the dump
neo4j-admin database load neo4j --from-path=./tfid-data --overwrite-destination=true

# 4. Start Neo4j
neo4j start
```

### Method 2: For Neo4j 4.x

```bash
# 1. Stop Neo4j if it's running
neo4j stop

# 2. Load the dump
neo4j-admin load --from=./tfid-data/neo4j.dump --database=neo4j --force

# 3. Start Neo4j
neo4j start
```

### Method 3: Using Docker

```bash
# 1. Stop the container if running
docker stop neo4j

# 2. Copy dump file to import directory
cp ./tfid-data/neo4j.dump $HOME/neo4j/import/

# 3. Load the dump
docker exec neo4j neo4j-admin database load neo4j \
    --from-path=/var/lib/neo4j/import \
    --overwrite-destination=true

# 4. Restart the container
docker restart neo4j
```

## Starting Neo4j

### Standard Installation

```bash
# Start Neo4j
neo4j start

# Check status
neo4j status

# Stop Neo4j
neo4j stop

# Restart Neo4j
neo4j restart
```

### Docker

```bash
# Start container
docker start neo4j

# Stop container
docker stop neo4j

# View logs
docker logs neo4j
```

## Accessing Neo4j Browser

Once Neo4j is running, you can access it through:

1. **Neo4j Browser**: Open your web browser and navigate to:
   ```
   http://localhost:7474
   ```

2. **Default Credentials**:
   - Username: `neo4j`
   - Password: `neo4j` (you'll be prompted to change this on first login)

3. **Verify the Database**:
   ```cypher
   // Count all nodes
   MATCH (n) RETURN count(n)

   // View node labels
   CALL db.labels()

   // View relationship types
   CALL db.relationshipTypes()
   ```

## Configuration

### Setting the Password

After first login, you'll need to set a new password. You can also set it via command line:

```bash
# Set password using neo4j-admin
neo4j-admin dbms set-initial-password your_new_password
```

### Configuration File

The main configuration file is located at:
- **macOS/Linux**: `/usr/local/etc/neo4j/neo4j.conf` or `$NEO4J_HOME/conf/neo4j.conf`
- **Windows**: `C:\Program Files\Neo4j\conf\neo4j.conf`

### Important Configuration Settings

Edit `neo4j.conf` to customize:

```properties
# Enable remote connections (uncomment to allow)
server.default_listen_address=0.0.0.0

# Increase memory for better performance
server.memory.heap.initial_size=1g
server.memory.heap.max_size=2g
server.memory.pagecache.size=1g

# Enable APOC procedures (if needed)
dbms.security.procedures.unrestricted=apoc.*
```

### Environment Variables for Python

Create a `.env` file in your project root:

```bash
# Neo4j Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# API Keys
GROQ_API_KEY=your_groq_api_key
SS_API_KEY=your_semantic_scholar_api_key
```

## Troubleshooting

### Issue: "Database does not exist"

**Solution**: Make sure you've loaded the dump correctly and the database name matches:

```bash
# List all databases
neo4j-admin database list

# If the database isn't listed, reload the dump
neo4j-admin database load neo4j --from-path=./tfid-data --overwrite-destination=true
```

### Issue: "Port 7474 or 7687 already in use"

**Solution**: Another Neo4j instance or application is using the port:

```bash
# Find process using the port
lsof -i :7474
lsof -i :7687

# Kill the process or change Neo4j ports in neo4j.conf
```

### Issue: "Insufficient memory"

**Solution**: Increase heap size in `neo4j.conf`:

```properties
server.memory.heap.initial_size=2g
server.memory.heap.max_size=4g
```

### Issue: "Authentication failed"

**Solution**: Reset the password:

```bash
# Stop Neo4j
neo4j stop

# Delete auth file
rm $NEO4J_HOME/data/dbms/auth

# Restart and set new password
neo4j start
neo4j-admin dbms set-initial-password your_new_password
```

### Issue: Docker container won't start

**Solution**: Check logs and permissions:

```bash
# View logs
docker logs neo4j

# Ensure directories have correct permissions
chmod -R 755 $HOME/neo4j/data
chmod -R 755 $HOME/neo4j/logs
```

## Verifying the Installation

Run these Cypher queries in Neo4j Browser to verify the data:

```cypher
// 1. Count nodes by type
MATCH (n)
RETURN labels(n) as NodeType, count(n) as Count
ORDER BY Count DESC

// 2. Count relationships by type
MATCH ()-[r]->()
RETURN type(r) as RelationType, count(r) as Count
ORDER BY Count DESC

// 3. Sample papers
MATCH (p:Paper)
RETURN p.title, p.year, p.citation_count
LIMIT 10

// 4. Check for semantic relationships
MATCH (citing:Paper)-[r:EXTENDS|OUTPERFORMS|ADAPTS|ANALYZES]->(cited:Paper)
RETURN citing.title, type(r), cited.title
LIMIT 5
```

## Performance Optimization

### Create Indexes

For better query performance, create indexes on frequently queried properties:

```cypher
// Create indexes
CREATE INDEX paper_id_idx IF NOT EXISTS FOR (p:Paper) ON (p.paper_id);
CREATE INDEX author_id_idx IF NOT EXISTS FOR (a:Author) ON (a.author_id);
CREATE INDEX paper_year_idx IF NOT EXISTS FOR (p:Paper) ON (p.year);
CREATE INDEX paper_title_idx IF NOT EXISTS FOR (p:Paper) ON (p.title);

// Verify indexes
SHOW INDEXES;
```

### Monitor Performance

```cypher
// View query performance
CALL dbms.listQueries();

// View database statistics
CALL apoc.meta.stats();
```

## Next Steps

After successfully setting up Neo4j and loading the database:

1. Return to the main [README.md](README.md) for usage instructions
2. Configure your `.env` file with the correct credentials
3. Install Python dependencies using Poetry
4. Start building or querying the knowledge graph

## Additional Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [Neo4j Browser Guide](https://neo4j.com/docs/browser-manual/current/)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
