# Webshop Infrastructure

This project uses Terraform to provision and manage the infrastructure for a webshop application on Google Cloud Platform.

## Resources

The following resources are created by this project:

- Google Cloud Run service: This service runs the webshop application.
- Google Cloud Scheduler job: This job is scheduled to run every minute and sends a GET request to the `scrape-data` endpoint of the Cloud Run service.

## Prerequisites

- Google Cloud Platform account
- Terraform installed

## Setup

1. Clone this repository.
2. Navigate to the project directory.
3. Run `terraform init` to initialize the Terraform workspace.
4. Run `terraform apply` to create the infrastructure.

## Usage

Once the infrastructure is created, you can access the webshop application by navigating to the URL of the Cloud Run service. The Cloud Scheduler job will automatically send a GET request to the `scrape-data` endpoint of the Cloud Run service every minute.

## Cleanup

To destroy the infrastructure, run `terraform destroy`.

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the [MIT License](LICENSE.md).