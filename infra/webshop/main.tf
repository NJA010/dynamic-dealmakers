# Enable apis
resource "google_project_service" "cloud_scheduler" {
  service = "cloudscheduler.googleapis.com"
}

# Create resources
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

resource "google_cloudbuild_trigger" "server-build-trigger" {
  location = "europe-west1"

  trigger_template {
    branch_name = "main"
    repo_name   = "github_nja010_dynamic-dealmakers"
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


# building evidence
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
