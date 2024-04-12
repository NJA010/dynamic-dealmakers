# enable services
resource "google_project_service" "cloud_run" {
  service = "run.googleapis.com"
}

# create resources
resource "google_cloud_run_v2_service" "dd-service" {
  name     = "dd-service"
  ingress  = "INGRESS_TRAFFIC_ALL"
  location = "europe-west1"

  template {
    containers {
      image = "europe-west1-docker.pkg.dev/dynamicdealmakers-7012254/dd-repo/dd_webshop:latest"

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

resource "google_cloud_run_v2_service" "dd-evidence-server" {
  name     = "dd-evidence-server"
  ingress  = "INGRESS_TRAFFIC_ALL"
  location = "europe-west1"

  template {
    containers {
      image = "europe-west1-docker.pkg.dev/dynamicdealmakers-7012254/dd-repo/evidence-server:latest"
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