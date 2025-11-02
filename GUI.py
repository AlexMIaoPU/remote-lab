from nicegui import ui
import base64
import asyncio
import os
import cv2
import numpy as np
from MainImageProcessingLogic import run_segmentation_on_image

# Global holders accessible from other modules
uploaded_image_bytes = None         # raw bytes
uploaded_image_cv2 = None           # cv2 image (BGR numpy array)

# Simple callback registry so other code can be notified
on_image_uploaded_callbacks = []

def register_image_callback(fn):
    """Register fn(bytes, cv2_image, filename) to be called after upload."""
    on_image_uploaded_callbacks.append(fn)

ui.label('Hello NiceGUI!')

# placeholder for the uploaded image
preview = ui.image('').style('max-width: 50%; max-height: 50vh;')
preview.visible = False

# processing indicator and control button (hidden until upload)
processing_label = ui.label('Processing...').style('font-weight: bold; color: red;')
processing_label.visible = False
process_button = ui.button('Run Segmentation', on_click=None)
process_button.visible = False

# output container for results
output_panel = ui.column().classes('gap-4')

def _np_rgb_to_data_url(img: np.ndarray) -> str:
    """Convert an RGB uint8 numpy image to a PNG data URL for ui.image."""
    if img is None:
        return ''
    # ensure uint8
    img_u8 = img.astype(np.uint8)
    # convert RGB -> BGR for OpenCV encoding
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', bgr)
    if not ok:
        return ''
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f'data:image/png;base64,{b64}'

async def handle_upload(e):
    # support both e.file (single) and e.files (list) events
    uploaded = getattr(e, 'file', None) or (e.files[0] if getattr(e, 'files', None) else None)
    if uploaded is None:
        ui.notify('No file uploaded')
        return

    # basic server-side type check (prefer content_type when available)
    content_type = getattr(uploaded, 'content_type', '') or ''
    name = getattr(uploaded, 'name', '') or ''
    ext = (name.split('.')[-1].lower() if name else '')
    allowed_exts = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp', 'tiff'}

    if content_type:
        if not content_type.startswith('image/'):
            ui.notify('Only image files are allowed')
            return
    else:
        if ext not in allowed_exts:
            ui.notify('Only image files are allowed')
            return

    # read file bytes (UploadFile.read is async)
    if hasattr(uploaded, 'read'):
        content = await uploaded.read()
    else:
        # fallback if content already present
        content = getattr(uploaded, 'content', b'')
        if callable(content):
            content = content()

    # store bytes in global variable for other modules
    global uploaded_image_bytes, uploaded_image_cv2
    uploaded_image_bytes = content

    # convert bytes -> cv2 image (BGR)
    arr = np.frombuffer(content, dtype=np.uint8)
    try:
        uploaded_image_cv2 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        uploaded_image_cv2 = None

    # optionally save to disk (uploads folder)
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    save_name = name or 'uploaded_image'
    save_path = os.path.join(uploads_dir, save_name)
    with open(save_path, 'wb') as f:
        f.write(content)

    # notify any registered callbacks
    for cb in on_image_uploaded_callbacks:
        try:
            cb(uploaded_image_bytes, uploaded_image_cv2, save_path)
        except Exception:
            pass

    # build data URL and show image
    if ext == 'jpg':
        ext = 'jpeg'
    if not ext:
        # try to infer from content_type
        if content_type.startswith('image/'):
            ext = content_type.split('/')[-1]
        else:
            ext = 'png'
    data_url = 'data:image/{};base64,{}'.format(ext, base64.b64encode(content).decode('utf-8'))

    preview.set_source(data_url)
    preview.visible = True
    ui.notify(f'Uploaded {name or "file"}')

    # show and enable the segmentation button after upload
    process_button.visible = True
    process_button.disabled = False

# set the input accept attribute via props and validate server-side as well
upload_comp = ui.upload(on_upload=handle_upload).props('accept="image/*"').classes('max-w-full')

# async handler for the segmentation button (accept optional event arg so it runs in the UI slot)
async def _run_segmentation_handler(e=None):
    import asyncio as _asyncio
    global uploaded_image_cv2
    if uploaded_image_cv2 is None:
        ui.notify('No image available for segmentation')
        return

    ui.notify('Segmentation starting...')
    # update button text and disable interactions
    process_button.text = 'Processing...'
    process_button.disabled = True
    upload_comp.disabled = True
    processing_label.visible = True

    try:
        # yield to event loop so frontend receives the text update before heavy work
        await asyncio.sleep(0.05)

        # call segmentation (run in thread if function is sync)
        if _asyncio.iscoroutinefunction(run_segmentation_on_image):
            seg_out_img, class_grid_im, res_results = await run_segmentation_on_image(uploaded_image_cv2)
        else:
            seg_out_img, class_grid_im, res_results = await _asyncio.to_thread(run_segmentation_on_image, uploaded_image_cv2)

        ui.notify('Segmentation finished')

        # Display all returned images in output_panel
        # Note: seg_out_img and class_grid_im are expected to be RGB numpy arrays
        if seg_out_img is not None:
            with output_panel:
                ui.label('Segmentation Output:')
                ui.image(_np_rgb_to_data_url(seg_out_img)).style('max-width: 60%;')

        if class_grid_im is not None:
            with output_panel:
                ui.label('Classified Grid:')
                ui.image(_np_rgb_to_data_url(class_grid_im)).style('max-width: 60%;')

        # For each ResistorProcessingResult, display image and print id/resistance
        for res in res_results:
            # res.image expected RGB
            img_url = _np_rgb_to_data_url(res.image) if getattr(res, 'image', None) is not None else ''
            with output_panel:
                ui.label(f'Resistor id: {res.id} — resistance: {res.resistance}')
                if img_url:
                    ui.image(img_url).style('max-width: 50%;')

        # optional: scroll to output panel (if you want)
        # output_panel.focus()   # uncomment if supported in your NiceGUI version

    except Exception as exc:
        ui.notify(f'Error during segmentation: {exc}')
    finally:
        # re-enable interactions
        process_button.text = 'Run Segmentation'
        processing_label.visible = False
        process_button.disabled = False
        upload_comp.disabled = False

# wire the button to the async handler so it executes in the UI slot (do NOT create a detached task)
process_button.on('click', _run_segmentation_handler)

ui.run()