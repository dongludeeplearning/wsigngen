mkdir -p body_models
cd body_models/

echo -e "The smpl files will be stored in the 'body_models/smplx/' folder\n"
gdown "https://drive.google.com/uc?id=1INYlGA76ak_cKGzvpOV2Pe6RkYTlXTW2"

unzip smpl.zip
echo -e "Cleaning\n"
rm smpl.zip

echo -e "Downloading done!"