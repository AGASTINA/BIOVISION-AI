import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError

def upload_to_s3(file_bytes: bytes, filename: str) -> bool:
    """
    Uploads an image to AWS S3 if credentials are provided in the environment.
    Returns True if upload succeeds, False if it fails or if credentials are missing.
    """
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    bucket_name = os.environ.get("S3_BUCKET_NAME")

    # If any essential configuration is missing, fallback to local storage
    if not all([aws_access_key, aws_secret_key, bucket_name]):
        return False
        
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        # Upload the file under the 'uploads/' prefix
        s3.put_object(Bucket=bucket_name, Key=f"uploads/{filename}", Body=file_bytes)
        return True
    except (BotoCoreError, ClientError) as e:
        # Silently fail so the main application never crashes if AWS is down
        print(f"AWS S3 Upload failed: {e}")
        return False
