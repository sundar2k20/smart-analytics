-- One-time setup: create a dedicated database for MLflow metadata.
-- Run as a Postgres superuser, e.g.:
--   psql -U postgres -f database/create_mlflow_db.sql
CREATE DATABASE mlflow;
