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

# store last segmentation results globally so other modules / handlers can access/edit
res_results_storage = []  # list of ResistorProcessingResult items returned by run_segmentation_on_image

def update_resistance(idx, new_value):
    """Update the stored resistance value for result at index idx; accepts numeric or string."""
    global res_results_storage
    try:
        item = res_results_storage[idx]
    except Exception:
        ui.notify(f'Failed to update resistance: invalid index {idx}')
        return
    # attempt to coerce to float, but fall back to raw string
    try:
        val = float(new_value) if new_value != '' else None
    except Exception:
        val = new_value
    try:
        if isinstance(item, dict):
            item['resistance'] = val
        else:
            setattr(item, 'resistance', val)
        ui.notify(f'Resistance updated for item {getattr(item, "id", idx)}')
    except Exception:
        ui.notify('Failed to set resistance on the item')

def register_image_callback(fn):
    """Register fn(bytes, cv2_image, filename) to be called after upload."""
    on_image_uploaded_callbacks.append(fn)


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


# async handler for the segmentation button (accept optional event arg so it runs in the UI slot)
async def _run_segmentation_handler(e=None):
    import asyncio as _asyncio
    global uploaded_image_cv2, res_results_storage
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

        # store results globally (replace previous)
        res_results_storage = list(res_results) if res_results is not None else []

        # clear previous output and redraw larger images and editable resistor entries
        try:
            output_panel.clear()
        except Exception:
            output_panel.items = []

        # Two-column layout: left -> main outputs (seg_out_img, class_grid_im),
        # right -> list of resistor cards (image + editable resistance)
        with output_panel:
            with ui.row().classes('items-start gap-8'):
                left_col = ui.column().style('flex: 0 0 520px; gap: 16px;')
                right_col = ui.column().style('flex: 1 1 auto; gap: 12px;')

                # Left column: show main images stacked and larger
                with left_col:
                    if seg_out_img is not None:
                        ui.label('Segmentation Output:')
                        ui.image(_np_rgb_to_data_url(seg_out_img)).style(
                            'width: 700px; height: auto; border: 1px solid #ddd; padding: 4px;'
                        )
                    if class_grid_im is not None:
                        ui.label('Classified Grid:')
                        ui.image(_np_rgb_to_data_url(class_grid_im)).style(
                            'width: 700px; height: auto; border: 1px solid #ddd; padding: 4px;'
                        )

                # Right column: one card per resistor with larger thumbnail and editable resistance
                for i, res in enumerate(res_results_storage):
                    with right_col:
                        with ui.card().style('padding: 12px; margin-bottom: 10px; display:flex; gap:12px; align-items:flex-start;'):
                            # thumbnail
                            img_url = _np_rgb_to_data_url(res.image) if getattr(res, 'image', None) is not None else ''
                            if img_url:
                                ui.image(img_url).style('width: 400px; height: auto; border: 1px solid #ccc;')
                            else:
                                ui.label('No image').style('width: 280px;')

                            # metadata + editable resistance
                            with ui.column():
                                ui.label(f'Resistor id: {getattr(res, "id", i)}')
                                curr_label = ui.label(f'Current resistance: {getattr(res, "resistance", "")}')
                                # on_change uses default arg to capture index correctly
                                inp = ui.input(
                                    value=str(getattr(res, 'resistance', '')),
                                    label='Edit resistance',
                                    on_change=lambda e, idx=i, lbl=curr_label: (
                                        update_resistance(idx, e.value),
                                        lbl.set_text(
                                            f'Current resistance: {((res_results_storage[idx]["resistance"] if isinstance(res_results_storage[idx], dict) else getattr(res_results_storage[idx], "resistance", e.value)))}'
                                        )
                                    )
                                )

    except Exception as exc:
        ui.notify(f'Error during segmentation: {exc}')
    finally:
        # re-enable interactions
        process_button.text = 'Run Segmentation'
        processing_label.visible = False
        process_button.disabled = False
        upload_comp.disabled = False


#############$$$$$$$$$$$$$$$$$$
##
## UI Setup
##
#############$$$$$$$$$$$$$$$$$$

ui.label('Hello There!!')

# top-row: left = controls (upload, button, processing), right = preview image
with ui.row().classes('items-start gap-8').style('align-items: flex-start;'):
    left_col = ui.column().style('flex: 0 0 360px; gap: 12px;')
    right_col = ui.column().style('flex: 1 1 auto; gap: 12px;')

    with left_col:
        ui.label('Controls').style('font-weight: bold;')
        # upload component placed on the left column
        upload_comp = ui.upload(on_upload=handle_upload).props('accept="image/*"').classes('max-w-full')
        # processing indicator and control button (hidden until upload)
        processing_label = ui.label('Processing...').style('font-weight: bold; color: red;')
        processing_label.visible = False
        process_button = ui.button('Run Segmentation', on_click=None)
        process_button.visible = False

    with right_col:
        ui.label('Preview').style('font-weight: bold;')
        preview = ui.image('').style('width: 500px; max-height: 70vh; border: 1px solid #ddd; padding: 6px;')
        preview.visible = False

# output container for results
output_panel = ui.column().classes('gap-4')

# wire the button to the async handler so it executes in the UI slot (do NOT create a detached task)
process_button.on('click', _run_segmentation_handler)

ui.run()