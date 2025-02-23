import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import os
import logging

class FirebaseStorageManager:
    def __init__(self, bucket_name):
        self.logger = logging.getLogger(__name__)
        
        # Validate service account file exists
        cred_path = 'firebase/service_account_key.json'
        if not os.path.exists(cred_path):
            raise FileNotFoundError(f"Firebase credentials file not found at {cred_path}")
            
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': bucket_name
                })
            self.bucket = storage.bucket(app=firebase_admin.get_app())
        except Exception as e:
            self.logger.error(f"Failed to initialize Firebase: {str(e)}")
            raise

    def upload_file(self, local_file_path, destination_blob_name):
        """Upload a file to Firebase Storage and return its public URL"""
        try:
            # Validate local file exists
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"Local file not found: {local_file_path}")
                
            # Upload file
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(local_file_path)
            
            # Make the blob publicly accessible
            blob.make_public()
            
            # Verify upload and get URL
            if not blob.exists():
                raise Exception("File upload failed - blob does not exist")
                
            return blob.public_url
            
        except Exception as e:
            self.logger.error(f"Failed to upload file {local_file_path}: {str(e)}")
            raise

    def delete_file(self, blob_name):
        """Delete a file from Firebase Storage"""
        try:
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                blob.delete()
                self.logger.info(f"Successfully deleted {blob_name}")
            else:
                self.logger.warning(f"File {blob_name} does not exist")
        except Exception as e:
            self.logger.error(f"Failed to delete file {blob_name}: {str(e)}")
            raise

# 'bucket' is an object defined in the google-cloud-storage Python library.
# See https://googlecloudplatform.github.io/google-cloud-python/latest/storage/buckets.html
# for more details.