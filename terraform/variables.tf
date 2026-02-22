variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "eu-central-1"  # Frankfurt — close to Germany, good for your narrative
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "ml-platform"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "ec2_instance_type" {
  description = "EC2 instance type for inference service"
  type        = string
  default     = "t2.micro"  # Free tier eligible
}

variable "dockerhub_image" {
  description = "Docker Hub image for inference service"
  type        = string
  default     = "your-dockerhub-username/rul-predictor:latest"
}