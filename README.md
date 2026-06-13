# Kuwait Civil ID OCR Extractor

A robust API for extracting information from Kuwait Civil ID and security card images using OCR technology. This service provides a secure, rate-limited REST API that processes images and extracts structured information such as names, civil ID, dates, nationality, address, serial number, and MRZ data.

![Bangladesh NID](https://i.imgur.com/example.png)

## 📋 Features

- **Template-Aware Extraction**: Uses 6 Kuwait card templates for front/back card layouts
- **Robust Text Extraction**: Uses EasyOCR with field-specific regions and fallback patterns
- **Multilingual OCR**: Supports English, Persian, and Arabic text by default
- **High Accuracy**: Crops known field regions before OCR when a template is detected
- **Secure API**: Token-based authentication and request rate limiting
- **Cross-Platform**: Works on both Windows and Linux environments
- **Field Validation**: Validates extracted information against provided data
- **Resource Management**: Efficient cleaning of temporary files
- **Comprehensive Logging**: Detailed logs for debugging and auditing

## 🔧 Requirements

- Python 3.8+ (tested on Python 3.12 and 3.13)
- Flask web framework
- OpenCV for image processing
- EasyOCR for text extraction
- Storage space for model files (~100MB)

## ⚙️ Installation

### Common Setup (All Platforms)

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/nid-ocr-extractor.git
   cd nid-ocr-extractor
   ```

2. **Create environment file:**

   ```bash
   # Copy example environment file
   cp .env.example .env

   # Generate secure tokens and update .env file
   python -c "import secrets; print(f'SECRET_KEY={secrets.token_hex(32)}')"
   python -c "import secrets; print(f'AUTH_TOKEN={secrets.token_hex(16)}')"
   ```

### Windows Setup

1. **Create and activate virtual environment:**

   ```powershell
   python -m venv win_venv
   win_venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt

   # Windows needs python-magic-bin instead of python-magic
   pip uninstall -y python-magic
   pip install python-magic-bin
   ```

### Linux Setup

1. **Create and activate virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install system dependencies:**

   For Debian/Ubuntu:

   ```bash
   sudo apt-get update
   sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libmagic1
   ```

   For Arch Linux:

   ```bash
   sudo pacman -Syu
   sudo pacman -S mesa glib2 libx11 libxext libxrender
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Starting the Server

```bash
# Start the Flask server
python app.py
```

By default, the server runs on `http://localhost:5000`.

### Testing with the Client

The repository includes a client script for testing the API:

```bash
# Basic usage with default settings
python client.py --token YOUR_AUTH_TOKEN

# Specify a custom image
python client.py --image path/to/id/image.jpg --token YOUR_AUTH_TOKEN

# Compare with known data
python client.py --name "John Doe" --dob "15 Mar 1985" --token YOUR_AUTH_TOKEN
```

### API Endpoints

#### `GET /`

Health check endpoint that confirms the API is running.

#### `POST /process_image`

Processes an ID card image and extracts information.

**Request Headers:**

- `X-API-Token`: Your authentication token from .env file

**Form Data:**

- `image`: The image file (JPEG, PNG)
- `template_id` (optional): One of the supported template IDs to skip automatic template detection
- `Name` (optional): Name for comparison
- `Date of Birth` (optional): Date of birth for comparison

**Response:**

```json
{
  "template_id": "kuwait_front",
  "card_type": "kuwaiti",
  "side": "front",
  "document_type": "Civil ID Card",
  "country": "Kuwait",
  "civil_id": "290022400724",
  "passport_no": "",
  "name": {
    "en": "NADER ASSAF SHUAIL ALOTAIBI",
    "ar": "نادر عساف شعيل العتيبي"
  },
  "nationality": {
    "code": "KWT",
    "text": "كويتي"
  },
  "sex": {
    "code": "M",
    "text": "ذكر"
  },
  "birth_date": "1990-02-24",
  "issue_date": "",
  "expiry_date": "2027-08-31",
  "blood_type": "",
  "profession": "",
  "serial_no": "",
  "mrz": {
    "line1": "",
    "line2": "",
    "line3": ""
  },
  "address": {
    "full": "",
    "governorate": "",
    "area": "",
    "block": "",
    "street": "",
    "building": "",
    "unit": "",
    "floor": "",
    "automated_address_no": "",
    "phone": "",
    "serial": ""
  },
  "similarity": {
    "status": "no_comparison_data_provided"
  }
}
```

Supported templates:

- `kuwait_front`: Kuwaiti civil ID front
- `kuwait_back`: Kuwaiti civil ID back
- `resident_front`: Resident civil ID front
- `resident_back`: Resident civil ID back
- `bedoon_front`: Bedoon/security card front
- `bedoon_back`: Bedoon/security card back

## ⚠️ Common Issues and Troubleshooting

### Windows Issues

1. **libmagic not found error:**

   ```
   ImportError: failed to find libmagic
   ```

   **Solution:** Replace `python-magic` with `python-magic-bin`:

   ```powershell
   pip uninstall -y python-magic
   pip install python-magic-bin
   ```

2. **DLL load failed error:**
   ```
   ImportError: DLL load failed while importing cv2
   ```
   **Solution:** Reinstall OpenCV:
   ```powershell
   pip uninstall -y opencv-python
   pip install opencv-python
   ```

### Linux Issues

1. **OpenGL/libGL.so.1 error:**

   ```
   ImportError: libGL.so.1: cannot open shared object file
   ```

   **Solution:** Install required libraries:

   ```bash
   # For Ubuntu/Debian
   sudo apt-get install -y libgl1-mesa-glx

   # For Arch Linux
   sudo pacman -S mesa
   ```

2. **Permission denied for cache directory:**
   ```
   PermissionError: [Errno 13] Permission denied: 'cache'
   ```
   **Solution:** Check permissions:
   ```bash
   chmod 750 cache
   ```

## 🛠️ Configuration

The application uses environment variables defined in .env for configuration:

| Variable           | Description                    | Default         |
| ------------------ | ------------------------------ | --------------- |
| SECRET_KEY         | Secret key for Flask           | Generated value |
| AUTH_TOKEN         | API authentication token       | Generated value |
| RATE_LIMIT         | Max requests per window        | 10              |
| RATE_LIMIT_WINDOW  | Rate limit window in seconds   | 60              |
| MAX_CONTENT_LENGTH | Max allowed file size in bytes | 5MB (5242880)   |
| CACHE_DIR          | Directory for temporary files  | cache           |
| OCR_LANGUAGES      | Comma-separated EasyOCR language codes | en,fa,ar |
| OCR_RECOG_NETWORK  | Optional EasyOCR recognition network override | Empty |
| OCR_INCLUDE_RAW_TEXT | Include raw OCR text under `debug.raw_text` | false |

By default, OCR runs with English, Persian, and Arabic enabled:

```env
OCR_LANGUAGES=en,fa,ar
```

Leave `OCR_RECOG_NETWORK` empty for multilingual OCR. Set it only when you intentionally want to force a specific EasyOCR recognition network.

## 🔒 Security Notes

1. Always use a strong, randomly generated AUTH_TOKEN
2. The API implements rate limiting to prevent abuse
3. Temporary files are automatically deleted after processing
4. Input validation helps prevent malicious uploads
5. Security headers mitigate common web vulnerabilities

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for the OCR engine
- Flask team for the web framework
- OpenCV contributors for image processing capabilities

---

**Note:** This software is intended for legitimate identity verification purposes. Please ensure compliance with local data protection and privacy regulations when handling personal identification information.
