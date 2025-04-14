
# MBOD Data Processor  

**Description**:  
The **MBOD Data Processor** is a repository for generating and managing datasets in a standardized HDF5 format. It includes tools for visualizing, verifying, and generating datasets, ensuring streamlined data preparation for downstream experiments and research.

---

## **Features**  
- Preprocess DICOM datasets into HDF5 format.  
- Convenient scripts for:  
  - **Dataset Visualization**: View random X-rays and label distributions.  
  - **Dataset Verification**: Check integrity and class support.  
  - **Dataset Generation**: Convert datasets into standardized HDF5 files.  
- Reusable utilities for transforming and managing datasets.  

---

## **Getting Started**  

### **Prerequisites**  
Ensure you have **Python 3.9+** installed.  

**Install `uv` for dependency management**:  
Even if you haven't tried it before, I strongly recommend using uv as your package manager, you can install it with:
```bash
pip install uv
```
It is extremely fast and robust. For more information, see the [uv Getting Started Guide](https://github.com/astral-sh/uv).  

---

### **Setup**  
To set up the repository, clone it and install dependencies:  
```bash
# Clone the repository
git clone https://github.com/yourusername/mbod-data-processor.git
cd mbod-data-processor
```
You can install requirements using requirements.txt with pip or uv:
```
# Install dependencies using uv or pip from requirements .txt
uv pip install -r requirements.txt
```
or
```
pip install -r requirements.txt
```
If you are using pip, you should probably set up a venv first, like you typically would.

uv also has another option, you can just run:
```
uv sync
```
In the directory, and all the requirements and venv will be created.

---

## **Usage**  

### **1. Generate the Merged MBOD Silicosis Dataset**  
Before generating the dataset you have to specify 4 configuration settings in the config.yaml:
```yaml
silicosis_v1:
  imgpath: "" # Path to V1 images
  csvpath: "" # Path to V1 csv
  delimeter: ","
  image_id_column: "Anonymized Study Instance UID"
  radiologist_findings_columns:
    - "Radiologist: Findings"
  profusion_score_column: "MBOD Panel: Profusion"

silicosis_v2:
  imgpath: "" # Path to V2 images
  csvpath: "" # Path to V2 csv
```
That is, the paths to the V1 and V2 image directory and .csv file, respectively. You may have to convert the excel files to .csv.

After those configuration settings are specified, you can process the two datasets into .hdf5 format using:
```bash
uv run generate.py
```

### **2. Verify Dataset Integrity**  
To count class support and check integrity:  
```bash
uv run verify.py
```

### **3. Visualize Dataset**  
To display random X-rays and label information:  
```bash
uv run visualise.py
```

---

## **Contributing**  
Contributions are welcome! Please open an issue first to discuss your proposed changes.

---

## **License**  
[MIT License](LICENSE)  

---

## **Resources**  
- **uv Documentation**: [uv on GitHub](https://github.com/astral-sh/uv)  
- **HDF5 Format**: [HDF Group Overview](https://www.hdfgroup.org/solutions/hdf5/)  
