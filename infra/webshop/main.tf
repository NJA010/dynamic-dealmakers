# Enable apis
resource "google_project_service" "cloud_run" {
  service = "run.googleapis.com"
}

resource "google_project_service" "cloud_scheduler" {
  service = "cloudscheduler.googleapis.com"
}

# Create resources
resource "google_cloud_run_v2_service" "dd-service" {
  name     = "dd-service"
  ingress  = "INGRESS_TRAFFIC_ALL"
  location = "europe-west1"

  template {
    containers {
      image = "europe-west1-docker.pkg.dev/dynamicdealmakers-7012254/dd-repo/dd_webshop:v0.0.18"

      env {
        name  = "TF_VAR_api_key"
        value = var.api_key
      }
    }
  }

  depends_on = [
    google_project_service.cloud_run,
  ]
}

resource "google_service_account" "scheduler" {
  account_id   = "scheduler"
  display_name = "Scheduler Service Account"
}

resource "google_cloud_run_service_iam_member" "scheduler_job_runner" {
  service = google_cloud_run_v2_service.dd-service.name
  project = "dynamicdealmakers-7012254"
  role    = "roles/run.invoker"
  member  = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_cloud_run_service_iam_member" "wesley" {
  service = google_cloud_run_v2_service.dd-service.name
  project = "dynamicdealmakers-7012254"
  role    = "roles/run.invoker"
  member  = "user:wboelrijk@xccelerated.io"
}

# whitelist all
resource "google_cloud_run_service_iam_member" "all" {
  service = google_cloud_run_v2_service.dd-service.name
  project = "dynamicdealmakers-7012254"
  role    = "roles/run.invoker"
  member  = "allUsers"
}

resource "google_cloud_scheduler_job" "dd-scheduler" {
  name     = "dd-scheduler"
  schedule = "*/1 * * * *"
  time_zone = "Europe/Amsterdam"
  region = "europe-west1"

  http_target {
    http_method = "GET"
    uri = "${google_cloud_run_v2_service.dd-service.uri}/scrape-data"
    oidc_token {
      service_account_email = google_service_account.scheduler.email
      audience              = google_cloud_run_v2_service.dd-service.uri  
    }
  }

  depends_on = [
    google_cloud_run_v2_service.dd-service,
  ]
}

resource "google_pubsub_topic" "evidence-trigger-topic" {
  name = "evidence-update-id"
}

resource "google_cloud_scheduler_job" "dd-evidence-update-scheduler" {
  name     = "dd-evidence-update-scheduler"
  schedule = "0 * * * *"
  time_zone = "Europe/Amsterdam"
  region = "europe-west1"

  pubsub_target {
    # topic.id is the topic's full resource name.
    topic_name = google_pubsub_topic.evidence-trigger-topic.id
    data       = base64encode("test")
    }

  depends_on = [
    google_cloud_run_v2_service.dd-service
  ]
}

resource "google_cloudbuild_trigger" "evidence-build-trigger" {
  location = "europe-west1"

  trigger_template {
    branch_name = "main"
    repo_name   = "github_nja010_dynamic-dealmakers"
  }

  pubsub_config {
      topic = google_pubsub_topic.evidence-trigger-topic.id
    }

  substitutions = {
    _EVIDENCE_CONNECTION_STRING = var.postgres_conn_string
  }

  filename = "cloudbuild_evidence.yaml"
  depends_on = [
    google_cloud_scheduler_job.evidence-update-scheduler,
  ]
}


resource "google_cloudbuild_trigger" "server-build-trigger" {
  location = "europe-west1"

  trigger_template {
    branch_name = "main"
    repo_name   = "github_nja010_dynamic-dealmakers"
  }

  pubsub_config {
      topic = google_pubsub_topic.evidence-trigger-topic.id
    }

  substitutions = {
    _EVIDENCE_CONNECTION_STRING = var.postgres_conn_string
    _API_KEY = var.api_key
    _PROJECT_ID = "dynamicdealmakers-7012254"
    _SECRET_NAME = var.secret_name_postgres
    _SECRET_NAME_SA = var.secret_name_sa
  }

  filename = "cloudbuild_server.yaml"
  depends_on = [
    google_cloud_scheduler_job.evidence-update-scheduler,
  ]
}


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
