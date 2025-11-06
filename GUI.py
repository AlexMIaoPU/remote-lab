from nicegui import ui
import base64
import asyncio
import os
import cv2
import numpy as np
from MainImageProcessingLogic import run_segmentation_on_image, run_dijkstra_on_grid, generate_netlist
# store row_count returned by segmentation
row_count_storage = None

# Global holders accessible from other modules
uploaded_image_bytes = None         # raw bytes
uploaded_image_cv2 = None           # cv2 image (BGR numpy array)

# Simple callback registry so other code can be notified
on_image_uploaded_callbacks = []

# store last segmentation results globally so other modules / handlers can access/edit
res_results_storage = []  # list of ResistorProcessingResult items returned by run_segmentation_on_image

# prevent re-entrant Dijkstra runs
dijkstra_running = False

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
async def perform_segmentation(image):
    """
    Run the heavy segmentation step and return results.
    This function does NOT modify any module-level globals — it returns the results instead.
    Returned: (seg_out_img, class_grid_im, res_results, row_count)
    """
    import asyncio as _asyncio
    # call segmentation (may be sync or async)
    if _asyncio.iscoroutinefunction(run_segmentation_on_image):
        seg_out_img, class_grid_im, res_results, row_count = await run_segmentation_on_image(image)
    else:
        seg_out_img, class_grid_im, res_results, row_count = await _asyncio.to_thread(run_segmentation_on_image, image)
    return seg_out_img, class_grid_im, res_results, row_count

async def _run_segmentation_handler(e=None):
    import asyncio as _asyncio
    # NOTE: this function no longer mutates module globals; it returns results to caller
    if uploaded_image_cv2 is None:
        ui.notify('No image available for segmentation')
        return None, None, None, None

    ui.notify('Segmentation starting...')
    process_button.text = 'Processing...'
    process_button.disabled = True
    upload_comp.disabled = True

    try:
        await asyncio.sleep(0.05)

        # run segmentation via the pure function that RETURNS results
        seg_out_img, class_grid_im, res_results, row_count = await perform_segmentation(uploaded_image_cv2)

        ui.notify('Segmentation finished')

        # build UI for outputs but DO NOT assign module globals here
        try:
            output_panel.clear()
        except Exception:
            output_panel.items = []

        with output_panel:
            with ui.row().classes('items-start gap-8'):
                left_col = ui.column().style('flex: 0 0 520px; gap: 16px;')
                right_col = ui.column().style('flex: 1 1 auto; gap: 16px;')

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

                with right_col:
                    ui.label('Individual component instances:')

                for i, res in enumerate(res_results or []):
                    with right_col:
                        with ui.card().style('padding: 12px; margin-bottom: 10px; display:flex; gap:12px; align-items:flex-start;'):
                            img_url = _np_rgb_to_data_url(res.image) if getattr(res, 'image', None) is not None else ''
                            if img_url:
                                ui.image(img_url).style('width: 350px; height: auto; border: 1px solid #ccc;')
                            else:
                                ui.label('No image').style('width: 350px;')

                            with ui.column():
                                ui.label(f'Resistor id: {getattr(res, "id", i)}')
                                curr_label = ui.label(f'Current resistance: {getattr(res, "resistance", "")}')
                                inp = ui.input(
                                    value=str(getattr(res, 'resistance', '')),
                                    label='Edit resistance',
                                    on_change=lambda e, idx=i, lbl=curr_label: (
                                        update_resistance(idx, e.value),
                                        lbl.set_text(
                                            f'Current resistance: {((res_results[idx]["resistance"] if isinstance(res_results[idx], dict) else getattr(res_results[idx], "resistance", e.value)))}'
                                        )
                                    )
                                )

        # return results to caller for external global assignment
        return seg_out_img, class_grid_im, res_results, row_count

    except Exception as exc:
        ui.notify(f'Error during segmentation: {exc}')
        return None, None, None, None

    finally:
        process_button.text = 'Run Segmentation'
        process_button.disabled = False
        upload_comp.disabled = False

async def _run_dijkstra_handler(e=None):
    global res_results_storage, row_count_storage, dijkstra_running

    # guard against re-entry
    if dijkstra_running:
        ui.notify('Dijkstra already running')
        return
    dijkstra_running = True

    # immediately disable the button to avoid duplicate click handling
    dijkstra_button.disabled = True
    dijkstra_button.text = 'Running...'

    # debug: show current state at handler start
    print(f"[DEBUG] _run_dijkstra_handler start -> res_results_storage type={type(res_results_storage)} length={len(res_results_storage) if res_results_storage is not None else 'None'} row_count={row_count_storage}")
    await asyncio.sleep(0.05)  # give UI a tiny moment

    try:
        # run the dijkstra function in a thread to avoid blocking the UI
        rest_results = await run_dijkstra_on_grid(res_results_storage, row_count_storage)
        
        ui.notify('Dijkstra finished')

        # run the Netlist file generation        
        netlist_file_str = generate_netlist(rest_results)

        ui.notify('Netlist generation finished')

        # show dijkstra results box and populate list
        dijkstra_box.visible = True
        try:
            dijkstra_output_panel.clear()
        except Exception:
            dijkstra_output_panel.items = []

        for i, rr in enumerate(rest_results):
            # rr may be an ResistorResult object
            plugged_nodes = ', '.join(str(gp.node_id) for gp in rr.plugged_gps) if getattr(rr, 'plugged_gps', None) else 'N/A'
            txt = f'Result {i}: resistor_id={rr.id}, plugged_nodes={plugged_nodes}'
            # add label into the column using context manager
            with dijkstra_output_panel:
                ui.label(txt)

        # populate netlist code panel
        with netlist_file_panel:
            ui.code(netlist_file_str).classes('w-full')

    except Exception as exc:
        ui.notify(f'Error: {exc}')
    finally:
        dijkstra_button.text = 'Run Dijkstra on Grid'
        dijkstra_button.disabled = False
        dijkstra_running = False


# wire the button to the async handler so it executes in the UI slot (do NOT create a detached task)
async def run_seg_and_store(e=None):
    """Call the segmentation handler, store returned results in module globals for other handlers, and enable Dijkstra UI."""
    ui.colors(primary='#555')
    global res_results_storage, row_count_storage
    seg_out_img, class_grid_im, res_results, row_count = await _run_segmentation_handler(e)
    # if segmentation failed or returned no results, do not overwrite globals
    if res_results is None:
        print("[DEBUG] run_seg_and_store: segmentation returned None")
        return

    # update global storage in-place to avoid rebinding issues across modules
    res_results_storage.clear()
    try:
        res_results_list = list(res_results)
    except Exception:
        res_results_list = [res_results]
    if res_results_list:
        res_results_storage.extend(res_results_list)
    row_count_storage = row_count

    # debug/log
    print(f"[DEBUG] run_seg_and_store: stored {len(res_results_storage)} results, row_count={row_count_storage}; first_item_type={type(res_results_storage[0]) if res_results_storage else 'N/A'}")
    ui.notify(f"Stored {len(res_results_storage)} resistor(s)")

    ui.colors(primary='#5898d4')

    # only enable Dijkstra button if we have at least one resistor result
    if len(res_results_storage) > 0:
        # ensure UI has a moment to process the storage update before enabling the button
        await asyncio.sleep(0.02)
        dijkstra_button.visible = True
        dijkstra_button.disabled = False
    else:
        dijkstra_button.visible = False
        dijkstra_button.disabled = True

#############$$$$$$$$$$$$$$$$$$
##
## UI Setup
##
#############$$$$$$$$$$$$$$$$$$

ui.label('UNSW Remote Lab System').style('color: #6E93D6; font-size: 300%; font-weight: 500')

# top-row: left = controls (upload, button, processing), right = preview image
with ui.row().classes('items-start gap-8').style('align-items: flex-start;'):
    left_col = ui.column().style('flex: 0 0 520px; gap: 12px;')
    right_col = ui.column().style('flex: 1 1 auto; gap: 12px;')

    with left_col:
        ui.label('Controls').style('font-weight: bold; font-size: 200%; font-weight: 500')
        # upload component placed on the left column
        upload_comp = ui.upload(on_upload=handle_upload).props('accept="image/*"').classes('max-w-full')
        # processing indicator and control button (hidden until upload)
        process_button = ui.button('Run Segmentation', on_click=None)
        process_button.visible = False
        # Dijkstra button placed below controls (initially hidden)
        dijkstra_button = ui.button('Run Dijkstra & Generate Netlist', on_click=None)
        dijkstra_button.visible = False
        dijkstra_button.disabled = True

    with right_col:
        ui.label('Image Preview').style('font-weight: bold; font-size: 200%; font-weight: 500')
        preview = ui.image('').style('width: 500px; max-height: 70vh; border: 1px solid #ddd; padding: 6px;')
        preview.visible = False
# separate area for dijkstra output
dijkstra_box = ui.card().style('margin-top: 12px; padding:12px;')
dijkstra_box.visible = False
with dijkstra_box:
    with ui.row().classes('items-start gap-8'):
        left_col = ui.column().style('flex: 0 0 520px; gap: 16px;')
        right_col = ui.column().style('flex: 1 1 auto; gap: 16px;')

        with left_col:
            ui.label('Dijkstra Results:').style('font-weight: bold;')
            # this column will be cleared / populated when results are ready
            dijkstra_output_panel = ui.column().classes('gap-2')

        with right_col:
            ui.label('Generated Netlist:').style('font-weight: bold;')
            netlist_file_panel = ui.column().classes('gap-2')

# output container for results
output_panel = ui.column().classes('gap-4')

process_button.on('click', run_seg_and_store)

# wire dijkstra button
dijkstra_button.on('click', _run_dijkstra_handler)

ui.run()