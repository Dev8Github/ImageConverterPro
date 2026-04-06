from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from nicegui import events, ui

APP_TITLE = 'ImageConvertPro NiceGUI Edition'
ADJUSTMENTS = [
    'Exposure', 'Contrast', 'Brightness', 'Saturation', 'Sharpness',
    'Highlights', 'Shadows', 'Whites', 'Blacks', 'Clarity',
    'Vibrance', 'Temperature', 'Tint', 'Hue', 'Fade', 'Dehaze',
    'Gamma', 'Vignette',
]


@dataclass
class UploadedImage:
    name: str
    content: bytes
    adjustments: Dict[str, int] = field(default_factory=lambda: {name: 50 for name in ADJUSTMENTS})

    @property
    def data_url(self) -> str:
        encoded = base64.b64encode(self.content).decode('ascii')
        return f'data:image/*;base64,{encoded}'


class NiceGUIState:
    def __init__(self) -> None:
        self.images: List[UploadedImage] = []
        self.selected: Optional[UploadedImage] = None
        self.preview = None
        self.adjustment_controls: Dict[str, Dict[str, object]] = {}
        self.info_label = None

    def set_preview(self, image: Optional[UploadedImage]) -> None:
        if self.preview is None:
            return
        if image is None:
            self.preview.set_source('')
            if self.info_label:
                self.info_label.set_text('Import images to start previewing')
            return
        self.preview.set_source(image.data_url)
        if self.info_label:
            self.info_label.set_text(f'{image.name} | Fit to screen preview')

    def select_image(self, image: Optional[UploadedImage]) -> None:
        self.selected = image
        self.set_preview(image)
        self.refresh_adjustments()
        render_files.refresh()

    def refresh_adjustments(self) -> None:
        if self.selected is None:
            return
        for name, controls in self.adjustment_controls.items():
            value = self.selected.adjustments[name]
            controls['slider'].value = value
            controls['number'].value = value


state = NiceGUIState()

ui.colors(primary='#4f46e5', secondary='#06b6d4', accent='#f59e0b', dark='#0f172a', positive='#10b981')
ui.add_head_html('''
<style>
body { background: linear-gradient(135deg, #f8fafc 0%, #dbeafe 100%); }
.nice-shell { max-width: 1500px; margin: 0 auto; }
.glass-card { background: rgba(255,255,255,0.78); backdrop-filter: blur(16px); border: 1px solid rgba(148,163,184,0.22); border-radius: 24px; box-shadow: 0 16px 40px rgba(15,23,42,0.10); }
.hero-title { font-size: 2rem; font-weight: 800; color: #0f172a; }
.hero-copy { color: #475569; }
.file-pill { border-radius: 14px; padding: 10px 14px; justify-content: flex-start; width: 100%; }
.adjust-card .q-field__control { border-radius: 14px; }
</style>
''')


def sync_adjustment(name: str, value: int) -> None:
    if state.selected is None:
        return
    value = max(0, min(100, int(value)))
    state.selected.adjustments[name] = value
    controls = state.adjustment_controls[name]
    if controls['slider'].value != value:
        controls['slider'].value = value
    if controls['number'].value != value:
        controls['number'].value = value


async def handle_upload(event: events.UploadEventArguments) -> None:
    image = UploadedImage(name=event.name, content=event.content.read())
    state.images.append(image)
    state.select_image(image)


def auto_adjust() -> None:
    if state.selected is None:
        ui.notify('Import or select an image first', color='warning')
        return
    preset = {
        'Exposure': 54,
        'Contrast': 52,
        'Brightness': 53,
        'Highlights': 42,
        'Shadows': 61,
        'Whites': 47,
        'Blacks': 56,
        'Clarity': 55,
        'Dehaze': 57,
        'Gamma': 48,
        'Vibrance': 54,
    }
    for key in ADJUSTMENTS:
        state.selected.adjustments[key] = preset.get(key, 50)
    state.refresh_adjustments()
    ui.notify('Auto adjustment preview applied', color='positive')


def reset_adjustments() -> None:
    if state.selected is None:
        return
    for key in ADJUSTMENTS:
        state.selected.adjustments[key] = 50
    state.refresh_adjustments()
    ui.notify('Adjustments reset', color='primary')


@ui.refreshable
def render_files() -> None:
    for image in state.images:
        classes = 'file-pill '
        classes += 'bg-indigo-50 text-indigo-900 ring-2 ring-indigo-400' if image is state.selected else 'bg-slate-100 text-slate-700 hover:bg-slate-200'
        ui.button(image.name, on_click=lambda img=image: state.select_image(img)).classes(classes).props('flat no-caps align=left')
    if not state.images:
        ui.label('Uploaded images will appear here.').classes('text-sm text-slate-500')


with ui.column().classes('nice-shell w-full gap-6 p-6'):
    with ui.row().classes('glass-card items-center justify-between w-full px-8 py-6'):
        with ui.column().classes('gap-1'):
            ui.label(APP_TITLE).classes('hero-title')
            ui.label('A playful browser-based NiceGUI workspace for file import, preview, adjustments, and export planning.').classes('hero-copy')
        with ui.row().classes('items-center gap-3'):
            ui.button('Auto', on_click=auto_adjust).props('unelevated color=primary')
            ui.button('Reset', on_click=reset_adjustments).props('outline color=primary')
            ui.button('Dark Mode').props('flat').on('click', lambda: ui.dark_mode().toggle())

    with ui.row().classes('w-full gap-6 items-stretch no-wrap'):
        with ui.column().classes('glass-card w-[290px] min-h-[760px] p-5 gap-4'):
            ui.label('Files').classes('text-xl font-bold text-slate-800')
            ui.upload(on_upload=handle_upload, auto_upload=True, multiple=True).props('accept=.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.gif,.ico,.icns,.heic,.heif').classes('w-full')
            with ui.column().classes('w-full gap-2'):
                render_files()

        with ui.column().classes('glass-card grow min-h-[760px] p-5 gap-4'):
            with ui.row().classes('items-center justify-between w-full'):
                ui.label('Preview').classes('text-xl font-bold text-slate-800')
                state.info_label = ui.label('Import images to start previewing').classes('text-sm text-slate-500')
            state.preview = ui.image('').classes('w-full h-[520px] rounded-3xl bg-slate-100 object-contain')
            with ui.row().classes('w-full gap-3'):
                ui.button('Fit to Screen').props('outline color=primary')
                ui.button('Before / After').props('outline color=secondary')
                ui.button('Export Selected').props('unelevated color=positive')

            with ui.row().classes('w-full gap-4'):
                with ui.column().classes('glass-card grow p-5 gap-3'):
                    ui.label('Export Formats').classes('text-lg font-semibold text-slate-800')
                    with ui.row().classes('gap-4'):
                        for fmt in ['JPG', 'PNG', 'WEBP', 'BMP', 'TIFF', 'ICO']:
                            ui.checkbox(fmt, value=fmt in {'JPG', 'PNG', 'WEBP'})
                with ui.column().classes('glass-card w-[240px] p-5 gap-3'):
                    ui.label('Target Size').classes('text-lg font-semibold text-slate-800')
                    ui.select(['No Limit', '100 KB', '500 KB', '1 MB', '5 MB', 'Custom'], value='1 MB').classes('w-full')
                    ui.number('Quality', value=85, min=10, max=100, step=1).classes('w-full')

        with ui.column().classes('glass-card w-[360px] min-h-[760px] p-5 gap-4 overflow-auto'):
            with ui.row().classes('items-center justify-between w-full'):
                ui.label('Adjustments').classes('text-xl font-bold text-slate-800')
                ui.badge('Editable values').props('color=primary outline')

            for adjustment in ADJUSTMENTS:
                with ui.column().classes('adjust-card gap-1 w-full rounded-2xl bg-slate-50 p-3'):
                    ui.label(adjustment).classes('text-sm font-semibold text-slate-700')
                    with ui.row().classes('items-center w-full gap-3 no-wrap'):
                        slider = ui.slider(min=0, max=100, value=50, step=1, on_change=lambda e, name=adjustment: sync_adjustment(name, e.value)).classes('grow')
                        number = ui.number(value=50, min=0, max=100, step=1, format='%d').classes('w-24')
                        number.on_value_change(lambda e, name=adjustment: sync_adjustment(name, e.value or 0))
                        state.adjustment_controls[adjustment] = {'slider': slider, 'number': number}

ui.run(title=APP_TITLE, reload=False, show=True)
