import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

cred = credentials.Certificate('path/to/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'PROJECT_ID.firebasestorage.app'
})

bucket = storage.bucket()
import os

class FirebaseStorageManager:
    def __init__(self, bucket_name):
        if not firebase_admin._apps:
            cred = credentials.Certificate('firebase/service_account_key.json')
            firebase_admin.initialize_app(cred, {
                'storageBucket': bucket_name
            })
        self.bucket = storage.bucket(app=firebase_admin.get_app())

    def upload_file(self, local_file_path, destination_blob_name):
        """Upload a file to Firebase Storage and return its public URL"""
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        
        # Make the blob publicly accessible
        blob.make_public()
        
        # Return the public URL
        return blob.public_url

    def delete_file(self, blob_name):
        """Delete a file from Firebase Storage"""
        blob = self.bucket.blob(blob_name)
        if blob.exists():
            blob.delete()

# 'bucket' is an object defined in the google-cloud-storage Python library.
# See https://googlecloudplatform.github.io/google-cloud-python/latest/storage/buckets.html
# for more details.