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

TEMPLATE_PROFILES = {
    'kuwait_front': {
        'card_type': 'kuwaiti',
        'side': 'front',
        'document_type': 'Civil ID Card',
        'regions': {
            'civil_id': (0.56, 0.17, 0.26, 0.09),
            'name_ar': (0.60, 0.25, 0.27, 0.13),
            'name_en': (0.24, 0.47, 0.45, 0.14),
            'nationality_code': (0.42, 0.68, 0.14, 0.08),
            'nationality_text': (0.66, 0.65, 0.16, 0.09),
            'sex_code': (0.42, 0.75, 0.10, 0.08),
            'sex_text': (0.68, 0.72, 0.12, 0.08),
            'birth_date': (0.48, 0.80, 0.23, 0.08),
            'expiry_date': (0.48, 0.88, 0.23, 0.08),
        },
    },
    'resident_front': {
        'card_type': 'resident',
        'side': 'front',
        'document_type': 'Civil ID Card',
        'regions': {
            'civil_id': (0.56, 0.17, 0.26, 0.09),
            'name_ar': (0.62, 0.24, 0.25, 0.13),
            'name_en': (0.24, 0.47, 0.45, 0.15),
            'passport_no': (0.51, 0.62, 0.24, 0.08),
            'nationality_code': (0.42, 0.69, 0.14, 0.08),
            'nationality_text': (0.66, 0.66, 0.16, 0.09),
            'sex_code': (0.42, 0.76, 0.10, 0.08),
            'sex_text': (0.68, 0.73, 0.12, 0.08),
            'birth_date': (0.48, 0.81, 0.23, 0.08),
            'expiry_date': (0.48, 0.89, 0.23, 0.08),
        },
    },
    'bedoon_front': {
        'card_type': 'bedoon',
        'side': 'front',
        'document_type': 'Security Card',
        'regions': {
            'civil_id': (0.57, 0.28, 0.27, 0.09),
            'name_ar': (0.58, 0.39, 0.28, 0.08),
            'nationality_text': (0.55, 0.48, 0.30, 0.08),
            'birth_date': (0.62, 0.57, 0.24, 0.08),
            'issue_date': (0.61, 0.67, 0.24, 0.08),
            'expiry_date': (0.61, 0.77, 0.24, 0.08),
        },
    },
    'kuwait_back': {
        'card_type': 'kuwaiti',
        'side': 'back',
        'document_type': 'Civil ID Card',
        'regions': {
            'civil_id': (0.58, 0.02, 0.26, 0.09),
            'blood_type': (0.18, 0.02, 0.13, 0.08),
            'block': (0.21, 0.09, 0.11, 0.07),
            'street': (0.41, 0.22, 0.18, 0.09),
            'building': (0.38, 0.30, 0.18, 0.08),
            'unit': (0.55, 0.24, 0.20, 0.09),
            'automated_address_no': (0.39, 0.29, 0.25, 0.10),
            'serial_no': (0.50, 0.48, 0.21, 0.08),
            'mrz': (0.02, 0.70, 0.96, 0.27),
        },
    },
    'resident_back': {
        'card_type': 'resident',
        'side': 'back',
        'document_type': 'Civil ID Card',
        'regions': {
            'civil_id': (0.58, 0.02, 0.26, 0.09),
            'blood_type': (0.23, 0.02, 0.12, 0.09),
            'profession': (0.54, 0.11, 0.36, 0.13),
            'block': (0.34, 0.20, 0.11, 0.08),
            'street': (0.71, 0.25, 0.20, 0.09),
            'building': (0.79, 0.35, 0.15, 0.08),
            'unit': (0.54, 0.35, 0.18, 0.08),
            'automated_address_no': (0.45, 0.41, 0.23, 0.09),
            'serial_no': (0.50, 0.49, 0.22, 0.08),
            'mrz': (0.02, 0.70, 0.96, 0.27),
        },
    },
    'bedoon_back': {
        'card_type': 'bedoon',
        'side': 'back',
        'document_type': 'Security Card',
        'regions': {
            'civil_id': (0.32, 0.08, 0.33, 0.08),
            'governorate': (0.63, 0.25, 0.30, 0.08),
            'area': (0.61, 0.33, 0.32, 0.08),
            'block': (0.28, 0.24, 0.22, 0.08),
            'building': (0.25, 0.33, 0.25, 0.08),
            'unit': (0.23, 0.44, 0.26, 0.08),
            'phone': (0.14, 0.62, 0.28, 0.08),
            'serial_no': (0.33, 0.79, 0.18, 0.08),
        },
    },
}


def empty_nid_data():
    return {
        'template_id': '',
        'card_type': '',
        'side': '',
        'document_type': '',
        'country': '',
        'civil_id': '',
        'passport_no': '',
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
        'blood_type': '',
        'profession': '',
        'serial_no': '',
        'mrz': {
            'line1': '',
            'line2': '',
            'line3': '',
        },
        'address': {
            'full': '',
            'governorate': '',
            'area': '',
            'block': '',
            'street': '',
            'building': '',
            'unit': '',
            'floor': '',
            'automated_address_no': '',
            'phone': '',
            'serial': '',
        },
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


def crop_relative_region(image, region):
    height, width = image.shape[:2]
    x, y, region_width, region_height = region
    left = max(0, int(x * width))
    top = max(0, int(y * height))
    right = min(width, int((x + region_width) * width))
    bottom = min(height, int((y + region_height) * height))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def read_region_text(ocr_reader, image, region):
    crop = crop_relative_region(image, region)
    if crop is None or crop.size == 0:
        return ''

    try:
        results = ocr_reader.readtext(
            crop,
            paragraph=False,
            detail=0,
            decoder='greedy',
            beamWidth=OCR_PARAMS.get("beamWidth", 5),
            contrast_ths=OCR_PARAMS.get("contrast_ths", 0.1),
            adjust_contrast=OCR_PARAMS.get("adjust_contrast", 0.5),
            text_threshold=OCR_PARAMS.get("text_threshold", 0.7),
            low_text=OCR_PARAMS.get("low_text", 0.4),
            link_threshold=OCR_PARAMS.get("link_threshold", 0.4),
        )
    except Exception as e:
        logger.warning(f"Region OCR failed: {e}")
        return ''

    return normalize_ocr_text(" ".join(str(item) for item in results)).strip()


def detect_template_id(normalized_text):
    text = normalized_text.upper()
    has_mrz = bool(re.search(r'\bIDKWT[A-Z0-9<]{8,}', text))
    if has_mrz:
        if re.search(r'المهنة|PROFESSION|SYR|IDKWTL', normalized_text, re.IGNORECASE):
            return 'resident_back'
        return 'kuwait_back'

    if re.search(r'المحافظة|رقم الهاتف|رقم التسلسل|المنطقة', normalized_text):
        return 'bedoon_back'
    if re.search(r'بطاقة|تاريخ الاصدار|تاريخ الإصدار|تاريخ الاصداء', normalized_text):
        return 'bedoon_front'

    if re.search(r'\bSTATE\s+OF\s+KUWAIT\b|\bCIVIL\s+ID\s+CARD\b', normalized_text, re.IGNORECASE):
        if re.search(r'\bPassport\s+No\b|\bSYR\b|\bEGY\b|\bIND\b|\bPAK\b|\bPHL\b', normalized_text, re.IGNORECASE):
            return 'resident_front'
        return 'kuwait_front'

    if re.search(r'\bPassport\s+No\b', normalized_text, re.IGNORECASE):
        return 'resident_front'
    if re.search(r'\bNationality\s+KWT\b|\bKWT\b', normalized_text, re.IGNORECASE):
        return 'kuwait_front'
    return ''


def apply_template_metadata(nid_data, template_id):
    if not template_id:
        return

    template = TEMPLATE_PROFILES.get(template_id, {})
    nid_data['template_id'] = template_id
    nid_data['card_type'] = template.get('card_type', '')
    nid_data['side'] = template.get('side', '')
    nid_data['document_type'] = template.get('document_type', nid_data['document_type'])
    nid_data['country'] = 'Kuwait'


def clean_region_value(text, labels):
    value = re.sub(r'\s+', ' ', text).strip(' :：.-ـ')
    for label in labels:
        value = re.sub(label, '', value, flags=re.IGNORECASE).strip(' :：.-ـ')
    return value


def first_date_in_text(text):
    date_text = first_group_match([r'(\d{1,4}[\/\.-]\d{1,2}[\/\.-]\d{1,4})'], text)
    return normalize_date_value(date_text) if date_text else ''


def merge_if_empty(container, key, value):
    if value and not container.get(key):
        container[key] = value


def apply_region_field(nid_data, field_name, text):
    if not text:
        return

    if field_name == 'civil_id':
        civil_id = clean_id_candidate(text)
        if civil_id:
            merge_if_empty(nid_data, 'civil_id', civil_id)
    elif field_name == 'passport_no':
        passport_no = first_group_match([r'([A-Z][A-Z0-9]{5,15})'], text, re.IGNORECASE)
        merge_if_empty(nid_data, 'passport_no', passport_no.upper())
    elif field_name == 'name_en':
        name = clean_english_name_candidate(clean_region_value(text, [r'\bName\b']))
        if name and re.search(r'[A-Z]', name, re.IGNORECASE):
            nid_data['name']['en'] = nid_data['name']['en'] or name.upper()
    elif field_name == 'name_ar':
        name = clean_name_candidate(clean_region_value(text, [r'الإ?سم', r'الأسم', r'الاستم', r'الإستم', r'اسم']))
        if name and re.search(r'[\u0600-\u06FF]', name):
            nid_data['name']['ar'] = nid_data['name']['ar'] or name
    elif field_name == 'nationality_code':
        nationality = first_group_match([r'\b([A-Z]{2,4})\b'], text, re.IGNORECASE)
        nid_data['nationality']['code'] = nid_data['nationality']['code'] or nationality.upper()
    elif field_name == 'nationality_text':
        nationality = clean_region_value(text, [r'الجنسية', r'الجنسیة'])
        if re.search(r'[\u0600-\u06FF]', nationality):
            nid_data['nationality']['text'] = nid_data['nationality']['text'] or nationality
    elif field_name == 'sex_code':
        sex = first_group_match([r'\b([MF])\b'], text, re.IGNORECASE)
        nid_data['sex']['code'] = nid_data['sex']['code'] or sex.upper()
    elif field_name == 'sex_text':
        sex = first_group_match([r'(ذكر|انثى|أنثى)'], text)
        nid_data['sex']['text'] = nid_data['sex']['text'] or sex
    elif field_name in {'birth_date', 'issue_date', 'expiry_date'}:
        merge_if_empty(nid_data, field_name, first_date_in_text(text))
    elif field_name == 'blood_type':
        blood_type = first_group_match([r'\b(AB|A|B|O)\s*([+-])\b'], text, re.IGNORECASE)
        sign = first_group_match([r'\b(?:AB|A|B|O)\s*([+-])\b'], text, re.IGNORECASE)
        if blood_type:
            nid_data['blood_type'] = nid_data['blood_type'] or f"{blood_type.upper()}{sign}"
    elif field_name == 'profession':
        merge_if_empty(nid_data, 'profession', clean_region_value(text, [r'المهنة']))
    elif field_name == 'serial_no':
        serial_no = clean_id_candidate(text)
        merge_if_empty(nid_data, 'serial_no', serial_no)
    elif field_name == 'mrz':
        mrz_lines = re.findall(r'[A-Z0-9<]{18,}', text.upper())
        for index, line in enumerate(mrz_lines[:3], start=1):
            nid_data['mrz'][f'line{index}'] = nid_data['mrz'][f'line{index}'] or line
    elif field_name in nid_data['address']:
        value = clean_region_value(text, [
            r'العنوان', r'المحافظة', r'المنطقة', r'القطعة', r'الشارع',
            r'المبنى', r'القسيمة', r'الوحدة', r'الدور', r'الرقم الآلي للعوان',
            r'الرقم الالي للعنوان', r'رقم الهاتف', r'رقم التسلسل',
        ])
        if field_name in {'block', 'building', 'unit', 'floor', 'automated_address_no', 'phone', 'serial'}:
            digits = clean_id_candidate(value)
            value = digits or value
        nid_data['address'][field_name] = nid_data['address'][field_name] or value


def apply_region_ocr(nid_data, ocr_reader, image, template_id):
    template = TEMPLATE_PROFILES.get(template_id)
    if not template or not isinstance(image, np.ndarray):
        return {}

    region_texts = {}
    for field_name, region in template.get('regions', {}).items():
        text = read_region_text(ocr_reader, image, region)
        region_texts[field_name] = text
        apply_region_field(nid_data, field_name, text)

    address_values = [
        value for key, value in nid_data['address'].items()
        if key != 'full' and value
    ]
    if address_values and not nid_data['address']['full']:
        nid_data['address']['full'] = ' '.join(address_values)

    return region_texts


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

def extract_nid_fields(image, template_id=None) -> dict:
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
            image_input = cv2.imread(image)
            if image_input is None:
                logger.error(f"Failed to read image using OpenCV: {image}")
                nid_data['error'] = "Invalid image provided"
                return nid_data
        elif isinstance(image, np.ndarray):
            # If image is already a numpy array
            logger.info("Processing image from numpy array")
            image_input = image
        else:
            logger.error(f"Unsupported image format: {type(image)}")
            nid_data['error'] = "Unsupported image format"
            return nid_data

        # Get the reader instance lazily
        ocr_reader = get_reader()
        logger.info("Starting OCR reading...")
        
        try:
            results = ocr_reader.readtext(
                image_input,
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

        if template_id and template_id not in TEMPLATE_PROFILES:
            logger.warning(f"Unknown template_id provided: {template_id}")
            template_id = None

        template_id = template_id or detect_template_id(normalized_text)
        apply_template_metadata(nid_data, template_id)
        region_texts = apply_region_ocr(nid_data, ocr_reader, image_input, template_id)
        if OCR_INCLUDE_RAW_TEXT and region_texts:
            nid_data.setdefault('debug', {})['template_regions'] = region_texts

        if re.search(r'\bSTATE\s+OF\s+KUWAIT\b|الكويت|کويت|کویعت', normalized_text, re.IGNORECASE):
            nid_data['country'] = nid_data['country'] or 'Kuwait'
        if re.search(r'\bCIVIL\s+ID\s+CARD\b|بطاقة|بطاقتا|مدنية|مدنیة', normalized_text, re.IGNORECASE):
            nid_data['document_type'] = nid_data['document_type'] or 'Civil ID Card'
        
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
                    nid_data['name'][name_key] = nid_data['name'][name_key] or name_candidate
                    logger.info(f"Found valid name: {name_candidate}")
                elif len(name_candidate) > 5 and not re.search(r'\d', name_candidate):
                    nid_data['name'][name_key] = nid_data['name'][name_key] or name_candidate
                    logger.info(f"Found potential single-word name: {name_candidate}")

        nid_data['nationality']['code'] = nid_data['nationality']['code'] or first_group_match(
            [r'\bNationality\s+([A-Z]{2,4})\b'],
            normalized_text,
            re.IGNORECASE,
        )
        nid_data['nationality']['text'] = nid_data['nationality']['text'] or first_group_match(
            [r'(?:الجنسية|الجنسیة)\s*[:：.]?\s*([\u0600-\u06FF]{2,30})'],
            normalized_text,
        )

        sex_code = first_group_match([r'\bSex\s+([MF])\b'], normalized_text, re.IGNORECASE)
        sex_text = first_group_match([r'(?:الجنس|الجنسى)\s*[:：.]?\s*(ذكر|انثى|أنثى)'], normalized_text)
        nid_data['sex']['code'] = nid_data['sex']['code'] or sex_code.upper()
        nid_data['sex']['text'] = nid_data['sex']['text'] or sex_text
        
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
                nid_data['birth_date'] = nid_data['birth_date'] or normalize_date_value(dob_match.group(1))
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
        nid_data['issue_date'] = nid_data['issue_date'] or (normalize_date_value(issue_date) if issue_date else '')
        nid_data['expiry_date'] = nid_data['expiry_date'] or (normalize_date_value(expiry_date) if expiry_date else '')
        
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
                    nid_data['civil_id'] = nid_data['civil_id'] or clean_id
                    logger.info(f"Found valid ID number format: {clean_id} (length: {len(clean_id)})")
                    break
                else:
                    # Even if length is unusual, keep it if it looks like an ID
                    nid_data['civil_id'] = nid_data['civil_id'] or clean_id
                    logger.info(f"Found ID with unusual length: {clean_id} (length: {len(clean_id)})")
                    # Continue searching for better matches
        
        logger.info(f"Extraction completed: {nid_data}")
        return nid_data
        
    except Exception as e:
        logger.exception(f"OCR processing failed: {str(e)}")
        nid_data['error'] = f"OCR processing failed: {str(e)}"
        return nid_data
