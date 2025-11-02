# XYZCare RAG - Heroku Deployment Guide

Complete guide for deploying XYZCare to Heroku with PostgreSQL and S3 storage.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Cost Estimate](#cost-estimate)
- [Architecture Overview](#architecture-overview)
- [Step 1: AWS S3 Setup](#step-1-aws-s3-setup)
- [Step 2: Prepare Local Data](#step-2-prepare-local-data)
- [Step 3: Git Configuration](#step-3-git-configuration)
- [Step 4: Create Heroku App](#step-4-create-heroku-app)
- [Step 5: Database Migration](#step-5-database-migration)
- [Step 6: Deploy Application](#step-6-deploy-application)
- [Step 7: Verify Deployment](#step-7-verify-deployment)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

---

## Prerequisites

Before starting, ensure you have:

1. **Heroku CLI** installed: https://devcenter.heroku.com/articles/heroku-cli
2. **Git** installed and configured
3. **AWS Account** with billing enabled
4. **OpenAI API Key** from https://platform.openai.com
5. **Local development environment** with Python 3.11+
6. **PDF manuals** prepared for ingestion

## Cost Estimate

**Recommended Configuration (Total: ~$12.50/month)**

- Heroku Basic Dyno: $7/month
- Heroku Postgres Mini: $5/month
- AWS S3 Storage (50 PDFs): ~$0.50/month
- OpenAI Embeddings (one-time): ~$0.25
- OpenAI STT (with local Whisper): $0/month

**Alternative with Cloud STT (Total: ~$32.50/month)**

- Heroku Eco Dyno: $5/month
- Heroku Postgres Mini: $5/month
- AWS S3: ~$0.50/month
- OpenAI Whisper API: ~$22/month (with rate limiting)

> **Recommendation**: Start with local Whisper STT to stay under $25/month budget.

---

## Architecture Overview

**Production Stack:**
- **Database**: Heroku Postgres with PostgreSQL full-text search
- **Storage**: AWS S3 for PDFs, FAISS indexes, and config files
- **STT**: Switchable between local Whisper or OpenAI API
- **Embeddings**: OpenAI text-embedding-3-small
- **Web Server**: FastAPI with uvicorn

**Key Changes from Local Development:**
- SQLite → PostgreSQL
- Local files → S3 storage
- Optional local Whisper support
- Environment-based configuration

---

## Step 1: AWS S3 Setup

### 1.1 Create S3 Bucket

```bash
# Install AWS CLI if not already installed
# macOS: brew install awscli
# Or download from: https://aws.amazon.com/cli/

# Configure AWS credentials
aws configure

# Create S3 bucket (choose a unique name)
aws s3 mb s3://xyzcare-manuals --region us-east-1
```

### 1.2 Create IAM User

```bash
# Create IAM user for app access
aws iam create-user --user-name xyzcare-app

# Attach S3 access policy
aws iam attach-user-policy \
  --user-name xyzcare-app \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create access keys
aws iam create-access-key --user-name xyzcare-app
```

**Save the output:**
- `AccessKeyId` → AWS_ACCESS_KEY_ID
- `SecretAccessKey` → AWS_SECRET_ACCESS_KEY

---

## Step 2: Prepare Local Data

### 2.1 Set Up Local Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2.2 Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` for local development:

```bash
USE_POSTGRES=false
USE_S3=false
STT_PROVIDER=openai
OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE
DATA_DIR=./data
```

### 2.3 Ingest Manuals Locally

```bash
# Place PDF files in data/manuals/
# Ensure data/alias_map.json is configured

# Run ingestion
python -m app.ingestion.ingest_manuals --rebuild
```

### 2.4 Upload Data to S3

```bash
# Upload PDFs
aws s3 sync data/manuals/ s3://xyzcare-manuals/manuals/ \
  --exclude ".*" --include "*.pdf"

# Upload config
aws s3 cp data/alias_map.json s3://xyzcare-manuals/config/

# Upload FAISS index with USE_S3=true
USE_S3=true python -m app.ingestion.ingest_manuals --upload-pdfs
```

---

## Step 3: Git Configuration

### 3.1 Configure Git Identity (New Computer)

```bash
# Set your name and email
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify
git config --global --list
```

### 3.2 Initialize Repository

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit - Heroku deployment ready"
```

---

## Step 4: Create Heroku App

### 4.1 Login and Create App

```bash
# Login to Heroku
heroku login

# Create app
heroku create xyzcare-rag
```

### 4.2 Add PostgreSQL Addon

```bash
# Add Heroku Postgres Mini ($5/month)
heroku addons:create heroku-postgresql:mini

# Wait for provisioning
heroku pg:wait

# Verify
heroku pg:info
```

### 4.3 Set Environment Variables

```bash
# Database and storage
heroku config:set USE_POSTGRES=true
heroku config:set USE_S3=true

# AWS credentials
heroku config:set AWS_ACCESS_KEY_ID=AKIA...
heroku config:set AWS_SECRET_ACCESS_KEY=...
heroku config:set AWS_REGION=us-east-1
heroku config:set S3_BUCKET_NAME=xyzcare-manuals

# OpenAI
heroku config:set OPENAI_API_KEY=sk-proj-...
heroku config:set STT_PROVIDER=whisper_local
heroku config:set EMBEDDING_MODEL=text-embedding-3-small

# Search config
heroku config:set FTS_TOP_K=20
heroku config:set ALIAS_MATCH_THRESHOLD=0.85

# Verify
heroku config
```

---

## Step 5: Database Migration

### 5.1 Run Schema Migration

```bash
# Connect to Heroku Postgres
heroku pg:psql

# Run schema (copy/paste from migrations/postgres_schema.sql)
# Or:
cat migrations/postgres_schema.sql | heroku pg:psql
```

### 5.2 Migrate Data

```bash
# Get database URL
DATABASE_URL=$(heroku config:get DATABASE_URL)

# Run migration
DATABASE_URL=$DATABASE_URL python migrations/migrate_sqlite_to_postgres.py
```

### 5.3 Verify Data

```bash
heroku pg:psql

SELECT COUNT(*) FROM manuals;
SELECT COUNT(*) FROM pages;
\q
```

---

## Step 6: Deploy Application

### 6.1 Deploy to Heroku

```bash
# Add Heroku remote
heroku git:remote -a xyzcare-rag

# Push to Heroku
git push heroku main

# Scale dyno
heroku ps:scale web=1
```

---

## Step 7: Verify Deployment

### 7.1 Test Endpoints

```bash
# Health check
curl https://YOUR_APP_NAME.herokuapp.com/healthz

# Test manual resolution
curl "https://YOUR_APP_NAME.herokuapp.com/api/manual/resolve?q=Pixel%209%20Pro"
```

---

## Troubleshooting

### Build Failures

**Slug size too large:**
```bash
# Switch to OpenAI STT to reduce size
heroku config:set STT_PROVIDER=openai
git commit --allow-empty -m "Reduce slug size"
git push heroku main
```

### Application Crashes

```bash
# Check logs
heroku logs --tail

# Restart
heroku restart
```

---

## Maintenance

### Adding New Manuals

```bash
# Upload to S3
aws s3 cp new_manual.pdf s3://xyzcare-manuals/manuals/

# Run ingestion
heroku run python -m app.ingestion.ingest_manuals
```

### Database Backups

```bash
# Create backup
heroku pg:backups:capture

# Download backup
heroku pg:backups:download
```

---

## Summary Checklist

- [ ] AWS S3 bucket created
- [ ] IAM user created with access keys
- [ ] PDFs uploaded to S3
- [ ] Git configured
- [ ] Heroku app created
- [ ] PostgreSQL addon added
- [ ] Environment variables set
- [ ] Database migrated
- [ ] Application deployed
- [ ] Endpoints tested

**Deployment Complete! 🎉**
