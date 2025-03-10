import kagglehub

# Download latest version
path = kagglehub.dataset_download("prathumarikeri/indian-sign-language-isl")

print("Path to dataset files:", path)