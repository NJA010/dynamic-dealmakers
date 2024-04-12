# secrets
resource "google_secret_manager_secret" "secret_name_postgres" {
  secret_id  = var.secret_name_postgres
  replication {
    user_managed {
      replicas {
        location = "europe-west1"
      }
    }
  }
}

resource "google_secret_manager_secret_version" "secret_name_postgres_version" {
  depends_on = [google_sql_database_instance.secret_name_postgres]
  secret     = google_secret_manager_secret.secret_name_postgres.id
  
  secret_data = jsonencode({
    host = var.pg_host,
    user = var.pg_user,
    password = var.pg_pass,
    database = "dynamic-dealmakers"
    }
  )
}
