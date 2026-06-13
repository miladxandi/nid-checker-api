import easyocr
import logging
import os
import re
import cv2
import numpy as np
from config import OCR_INCLUDE_RAW_TEXT, OCR_LANGUAGES, OCR_PARAMS, OCR_RECOG_NETWORK

# Configure logging for this module
logger = logging.getLogger(__name__)

# Lazy-loaded reader instance to avoid multiple initializations
reader = None

DIGIT_TRANSLATION_TABLE = str.maketrans({
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
})

ID_OCR_TRANSLATION_TABLE = str.maketrans({
    ')': '2',
    '(': '2',
    'O': '0',
    'o': '0',
    'I': '1',
    'l': '1',
})

ARABIC_FIELD_BOUNDARIES = (
    'تاریخ', 'تاريخ', 'تولد', 'میلاد', 'الميلاد', 'شماره', 'رقم',
    'کد', 'كد', 'الرقم', 'المدني', 'الجنسية', 'الجنسیة',
    'تعديل', 'مطالب', 'الاصدار', 'الإصدار', 'الاصداء', 'الانتهاء',
)
ARABIC_NAME_JUNK_WORDS = {'داا'}


def empty_nid_data():
    return {
        'document_type': '',
        'country': '',
        'civil_id': '',
        'name': {
            'en': '',
            'ar': '',
        },
        'nationality': {
            'code': '',
            'text': '',
        },
        'sex': {
            'code': '',
            'text': '',
        },
        'birth_date': '',
        'issue_date': '',
        'expiry_date': '',
    }


def normalize_ocr_text(text):
    """Normalize Arabic/Persian OCR text without losing the original script."""
    return text.translate(DIGIT_TRANSLATION_TABLE).replace('\u200c', ' ')


def clean_name_candidate(name_candidate):
    name_candidate = re.sub(r'\s+', ' ', name_candidate).strip(' :：.-ـ')
    for boundary in ARABIC_FIELD_BOUNDARIES:
        name_candidate = re.split(rf'\s+{re.escape(boundary)}\b', name_candidate, maxsplit=1)[0]

    words = name_candidate.split()
    while len(words) > 2 and words[-1] in ARABIC_NAME_JUNK_WORDS:
        words.pop()
    return ' '.join(words)


def clean_id_candidate(id_text):
    normalized_id = id_text.translate(ID_OCR_TRANSLATION_TABLE)
    return re.sub(r'\D', '', normalized_id)


def clean_english_name_candidate(name_candidate):
    name_candidate = re.sub(r'\s+', ' ', name_candidate).strip(' :：.-')
    return re.split(
        r'\s+(?:Nationality|Sex|Birth|Expiry|Civil|ID|No)\b',
        name_candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()


def normalize_date_value(date_text):
    parts = re.split(r'[\/\.-]', date_text.strip())
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return date_text.strip()

    first, second, third = parts
    if len(first) == 4:
        year, month, day = first, second, third
    elif len(third) == 4:
        year = third
        if int(first) > 12:
            day, month = first, second
        elif int(second) > 12:
            month, day = first, second
        else:
            day, month = first, second
    else:
        return date_text.strip()

    return f"{year.zfill(4)}-{month.zfill(2)}-{day.zfill(2)}"


def first_group_match(patterns, text, flags=0):
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return match.group(1).strip()
    return ''


def build_reader(languages, gpu_available):
    reader_options = {
        'gpu': gpu_available,
        'model_storage_directory': 'model',
        'download_enabled': True,
    }
    if OCR_RECOG_NETWORK:
        reader_options['recog_network'] = OCR_RECOG_NETWORK
    return easyocr.Reader(languages, **reader_options)


def get_reader():
    """Lazy load the OCR reader to avoid multiple initializations"""
    global reader
    if reader is None:
        languages = OCR_LANGUAGES or ['en']
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            reader = build_reader(languages, gpu_available)
            logger.info(
                f"Initialized EasyOCR for {languages} using "
                f"{'GPU' if gpu_available else 'CPU'}"
            )
        except Exception:
            logger.exception("Error initializing EasyOCR with GPU; falling back to CPU")
            reader = build_reader(languages, False)
    return reader

def extract_nid_fields(image) -> dict:
    """
    Extract and validate NID fields from the given image using OCR.
    Returns a dictionary with the extracted information.
    """
    nid_data = empty_nid_data()

    try:
        # Handle different image input formats
        if isinstance(image, str):
            # If image is a file path
            if not os.path.exists(image):
                logger.error(f"Image file not found: {image}")
                nid_data['error'] = "Image file not found"
                return nid_data
            logger.info(f"Processing image from path: {image}")
        elif isinstance(image, np.ndarray):
            # If image is already a numpy array
            logger.info("Processing image from numpy array")
        else:
            logger.error(f"Unsupported image format: {type(image)}")
            nid_data['error'] = "Unsupported image format"
            return nid_data

        # Get the reader instance lazily
        ocr_reader = get_reader()
        logger.info("Starting OCR reading...")
        
        try:
            results = ocr_reader.readtext(
                image,
                paragraph=True,
                detail=1,
                decoder='greedy',
                beamWidth=OCR_PARAMS.get("beamWidth", 5),
                contrast_ths=OCR_PARAMS.get("contrast_ths", 0.1),
                adjust_contrast=OCR_PARAMS.get("adjust_contrast", 0.5),
                text_threshold=OCR_PARAMS.get("text_threshold", 0.7),
                low_text=OCR_PARAMS.get("low_text", 0.4),
                link_threshold=OCR_PARAMS.get("link_threshold", 0.4),
            )
            logger.info(f"OCR reading completed, found {len(results)} text blocks")
        except Exception as e:
            logger.exception(f"Error during OCR reading: {e}")
            nid_data['error'] = f"OCR reading failed: {str(e)}"
            return nid_data
        
        # Process the results to extract text
        if not results:
            logger.warning("No text detected in the image")
            nid_data['error'] = "No text detected in the image"
            return nid_data
        
        # Extract full text from OCR results
        text_blocks = []
        for result in results:
            try:
                if len(result) >= 2:  # As long as we have bbox and text
                    text = result[1]
                    text_blocks.append(text)
            except Exception as e:
                logger.exception(f"Error processing OCR result block: {e}")
                continue
        
        full_text = " ".join(text_blocks)
        normalized_text = normalize_ocr_text(full_text)
        if OCR_INCLUDE_RAW_TEXT:
            nid_data['debug'] = {'raw_text': full_text.strip()}

        if re.search(r'\bSTATE\s+OF\s+KUWAIT\b|الكويت|کويت|کویعت', normalized_text, re.IGNORECASE):
            nid_data['country'] = 'Kuwait'
        if re.search(r'\bCIVIL\s+ID\s+CARD\b|بطاقة|بطاقتا|مدنية|مدنیة', normalized_text, re.IGNORECASE):
            nid_data['document_type'] = 'Civil ID Card'
        
        # IMPROVED NAME PATTERNS
        name_patterns = [
            # Match English names on Kuwait/Gulf ID cards.
            r'Name\s+([A-Z][A-Z\s]{2,80}?)(?=\s+(?:Nationality|Sex|Birth|Expiry|Civil|ال|KWT|\d)|$)',

            # Match Arabic names on Gulf ID cards, including common OCR variants.
            r'(?:الإ?سم|الأسم|الاستم|الإستم|اسم|الاسم)\s*[:：.]?\s*([\u0600-\u06FF][\u0600-\u06FF\sـ]{2,80}?)(?=\s+(?:Name|Nationality|Sex|Birth|الجنسية|الجنسیة|تاريخ|تاریخ|مطالب|تعديل|الرقم|رقم|\d)|$)',

            # Match Persian/Arabic name labels.
            r'(?:نام(?:\s+و\s+نام\s+خانوادگی)?|اسم|الاسم)\s*[:：.]?\s*([\u0600-\u06FF][\u0600-\u06FF\sـ]{2,80}?)(?=\s+(?:Name|Nationality|Sex|Birth|تاریخ|تاريخ|تولد|میلاد|الميلاد|شماره|رقم|کد|كد|الجنسية|الجنسیة|ID|NID|\d)|$)',

            # Match "Name" followed by uppercase name (common on ID cards)
            r'Name\s*[:.]?\s*([A-Z][A-Z\s\.]+)(?=\s+(?:fet|faot|ent|Date|Birth|DOB|NID|ID|No|\d)|\n|$)',
            
            # Match "Name:" label with very strict boundary
            r'Name\s*[:.]?\s+([A-Za-z][A-Za-z\s\.]{2,30})(?=\s+(?:fet|faot|ent|Date|Birth|DOB|NID|ID|No|\d)|\n|$)',
            
            # Common Bangladesh name format with "MD" or "Md." prefix
            r'\bM[dD]\.?\s+([A-Za-z]+\s+[A-Za-z]+(?:\s+[A-Za-z]+)?)\b',
            
            # Match all-caps names which are common on IDs (with stricter boundaries)
            r'Name\s*[:.]?\s*([A-Z]+\s+[A-Z]+(?:\s+[A-Z]+)?)\b',
        ]
        
        # Blacklist of phrases that should never be considered names
        name_blacklist = [
            "NATIONAL ID CARD", "ID CARD", "BANGLADESH", "GOVERNMENT", 
            "PEOPLES", "REPUBLIC", "CARD", "NATIONAL", "DATE OF BIRTH",
            "GOVERMENT", "soeezledt", "offthe", "Republic",
            "کارت ملی", "جمهوری", "تاریخ تولد", "تاريخ الميلاد", "شماره ملی",
            "الجنسية", "الجنسیة", "الرقم المدني"
        ]
        
        for pattern in name_patterns:
            name_match = re.search(pattern, normalized_text)  # Remove IGNORECASE flag
            if name_match:
                raw_name_candidate = name_match.group(1)
                if re.search(r'[\u0600-\u06FF]', raw_name_candidate):
                    name_candidate = clean_name_candidate(raw_name_candidate)
                    name_key = 'ar'
                else:
                    name_candidate = clean_english_name_candidate(raw_name_candidate)
                    name_key = 'en'
                
                # Skip if name is in blacklist
                if any(blacklisted.lower() in name_candidate.lower() for blacklisted in name_blacklist):
                    logger.info(f"Skipping blacklisted name: {name_candidate}")
                    continue
                    
                # Validate name has reasonable length and format
                if ' ' in name_candidate and 4 <= len(name_candidate) <= 50:
                    nid_data['name'][name_key] = name_candidate
                    logger.info(f"Found valid name: {name_candidate}")
                elif len(name_candidate) > 5 and not re.search(r'\d', name_candidate):
                    nid_data['name'][name_key] = name_candidate
                    logger.info(f"Found potential single-word name: {name_candidate}")

        nid_data['nationality']['code'] = first_group_match(
            [r'\bNationality\s+([A-Z]{2,4})\b'],
            normalized_text,
            re.IGNORECASE,
        )
        nid_data['nationality']['text'] = first_group_match(
            [r'(?:الجنسية|الجنسیة)\s*[:：.]?\s*([\u0600-\u06FF]{2,30})'],
            normalized_text,
        )

        sex_code = first_group_match([r'\bSex\s+([MF])\b'], normalized_text, re.IGNORECASE)
        sex_text = first_group_match([r'(?:الجنس|الجنسى)\s*[:：.]?\s*(ذكر|انثى|أنثى)'], normalized_text)
        nid_data['sex']['code'] = sex_code.upper()
        nid_data['sex']['text'] = sex_text
        
        # Extract date of birth with multiple patterns
        dob_patterns = [
            r'(?:Birth\s+Date|Birthdate)[:：.]?\s*(\d{1,4}[\/\.-]\d{1,2}[\/\.-]\d{1,4})',
            r'(?:تاریخ تولد|تاريخ الميلاد|تولد|الميلاد|DOB|Birth)[:：.]?\s*(\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2})',
            r'(?:تاریخ تولد|تاريخ الميلاد|تولد|الميلاد|DOB|Birth)[:：.]?\s*(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4})',
            r'(?:Date of Birth|DOB|Birth)[:.]?\s*(\d{1,2}[\s-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s-]\d{2,4})',
            r'(?:Date of Birth|DOB|Birth)[:.]?\s*(\d{1,2}[\/\.-]\d{1,2}[\/\.-]\d{2,4})',
            r'(\d{4}[\/\.-]\d{1,2}[\/\.-]\d{1,2})',
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'(\d{1,2}[\s-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\s-]\d{2,4})'
        ]
        
        for pattern in dob_patterns:
            dob_match = re.search(pattern, normalized_text, re.IGNORECASE)
            if dob_match:
                nid_data['birth_date'] = normalize_date_value(dob_match.group(1))
                logger.info(f"Found date of birth: {nid_data['birth_date']}")
                break

        issue_date = first_group_match(
            [
                r'(?:Issue\s+Date|Issued\s+Date)[:：.]?\s*(\d{1,4}[\/\.-]\d{1,2}[\/\.-]\d{1,4})',
                r'(?:تاريخ الاصدار|تاريخ الإصدار|تاريخ الاصداء)[:：.]?\s*(\d{1,4}[\/\.-]\d{1,2}[\/\.-]\d{1,4})',
            ],
            normalized_text,
            re.IGNORECASE,
        )
        expiry_date = first_group_match(
            [
                r'(?:Expiry\s+Date|Expiration\s+Date|Expiry)[:：.]?\s*(\d{1,4}[\/\.-]\d{1,2}[\/\.-]\d{1,4})',
                r'(?:تاريخ الانتهاء|تاریخ الانتهاء|انتهاء)[:：.]?\s*(\d{1,4}[\/\.-]\d{1,2}[\/\.-]\d{1,4})',
            ],
            normalized_text,
            re.IGNORECASE,
        )
        nid_data['issue_date'] = normalize_date_value(issue_date) if issue_date else ''
        nid_data['expiry_date'] = normalize_date_value(expiry_date) if expiry_date else ''
        
        # COMPREHENSIVE ID NUMBER PATTERNS
        id_patterns = [
            # Match English civil ID labels.
            r'(?:Civil\s+ID\s+No|Civil\s+ID|ID\s+No)[:：.]?\s*([^\u0600-\u06FFA-Za-z]{5,24})',

            # Match Kuwait/Gulf civil ID labels and tolerate OCR punctuation in digits.
            r'(?:الرقم\s+المدني|الرقم\s+المدنى|رقم\s+مدني|رقم\s+مدنى)[:：.]?\s*([^\u0600-\u06FF]{5,24})',

            # Match Persian/Arabic national ID labels
            r'(?:شماره(?:\s+ملی)?|کد(?:\s+ملی)?|كد(?:\s+وطني)?|رقم(?:\s+الهوية)?|رقم(?:\s+وطني)?)[:：.]?\s*([^\u0600-\u06FF]{5,24})',

            # Match "ID NO:" format 
            r'ID\s*NO[:.]?\s*(\d[\d\s-]{5,18}\d)',
            
            # Match "NID No" format
            r'NID\s*No[:.]?\s*(\d[\d\s-]{5,18}\d)',
            
            # Match exact Bangladesh NID format with spaces
            r'\b(\d{3}\s+\d{3}\s+\d{4})\b',
            
            # Match exact Bangladesh NID format with no spaces
            r'\b(\d{10}|\d{13}|\d{17})\b',
            
            # Match NID in machine-readable zone format
            r'[<I]BGD(\d{9,})[<\d]',
            
            # Match any format with explicit ID label
            r'(?:ID|NID|Number|No)[:.]\s*(\d[\d\s-]+\d)',
            
            # Match numbers with spaces or dashes
            r'\b(\d{3}[\s-]?\d{3}[\s-]?\d{4})\b',
            
            # Last resort - match any 10+ digit sequence
            r'\b(\d{10,})\b'
        ]
        
        for pattern in id_patterns:
            id_match = re.search(pattern, normalized_text)
            if id_match:
                id_text = id_match.group(1)
                clean_id = clean_id_candidate(id_text)
                if not clean_id:
                    continue
                
                # Validate: Bangladesh NIDs are typically 10, 13, or 17 digits
                if len(clean_id) in [10, 13, 17]:
                    nid_data['civil_id'] = clean_id
                    logger.info(f"Found valid ID number format: {clean_id} (length: {len(clean_id)})")
                    break
                else:
                    # Even if length is unusual, keep it if it looks like an ID
                    nid_data['civil_id'] = clean_id
                    logger.info(f"Found ID with unusual length: {clean_id} (length: {len(clean_id)})")
                    # Continue searching for better matches
        
        logger.info(f"Extraction completed: {nid_data}")
        return nid_data
        
    except Exception as e:
        logger.exception(f"OCR processing failed: {str(e)}")
        nid_data['error'] = f"OCR processing failed: {str(e)}"
        return nid_data
