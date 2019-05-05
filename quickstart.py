# import os
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from oauth2client.client import GoogleCredentials
#
# # 1. Authenticate and create the PyDrive client.
# gauth = GoogleAuth()
# gauth.LocalWebserverAuth()
#
# drive = GoogleDrive(gauth)
#
# # choose a local (colab) directory to store the data.
# local_download_path = 'AdvancedMachineLearning-Project/Dataset/'
# try:
#     os.makedirs(local_download_path)
# except: pass
#
# # 2. Auto-iterate using the query syntax
# #    https://developers.google.com/drive/v2/web/search-parameters
# file_list = drive.ListFile(
#     {'q': "'0B7EVK8r0v71pWGplNFhjc01NbzQ' in parents"}).GetList()  #use your own folder ID here
#
# for f in file_list:
#     # 3. Create & download by id.
#     print('title: %s, id: %s' % (f['title'], f['id']))
#     fname = f['title']
#     print('downloading to {}'.format(fname))
#     f_ = drive.CreateFile({'id': f['id']})
#     f_.GetContentFile(fname)

from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='0B7EVK8r0v71pWGplNFhjc01NbzQ', dest_path='/Dataset/dataset.zip', unzip=True)

