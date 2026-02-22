output "ec2_public_ip" {
  description = "Public IP of the inference EC2 instance"
  value       = aws_instance.inference.public_ip
}

output "ec2_public_dns" {
  description = "Public DNS of the inference EC2 instance"
  value       = aws_instance.inference.public_dns
}

output "s3_bucket_name" {
  description = "S3 bucket for ML artifacts"
  value       = aws_s3_bucket.ml_artifacts.bucket
}

output "inference_api_url" {
  description = "URL to access the inference API"
  value       = "http://${aws_instance.inference.public_ip}:8000"
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}