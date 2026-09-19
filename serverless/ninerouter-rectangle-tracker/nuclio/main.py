import base64
import io
import json

from PIL import Image

from model_handler import ModelHandler


def init_context(context):
    context.logger.info("Initializing 9Router Rectangle Tracker")
    context.user_data.model = ModelHandler()
    context.logger.info("9Router Rectangle Tracker ready")


def handler(context, event):
    data = event.body or {}
    image_b64 = data.get("image")
    if not image_b64:
        return context.Response(
            body=json.dumps({"error": "missing image"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    except Exception as exc:
        return context.Response(
            body=json.dumps({"error": f"invalid image: {type(exc).__name__}"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    shapes = list(data.get("shapes") or [])
    states = list(data.get("states") or [])
    count = max(len(shapes), len(states))
    if count == 0:
        return context.Response(
            body=json.dumps({"shapes": [], "states": []}),
            headers={},
            content_type="application/json",
            status_code=200,
        )

    if len(shapes) < count:
        shapes.extend([None] * (count - len(shapes)))
    if len(states) < count:
        states.extend([None] * (count - len(states)))

    output_shapes = []
    output_states = []

    for idx in range(count):
        shape, state = context.user_data.model.infer(
            image=image,
            shape=shapes[idx],
            state=states[idx],
        )
        output_shapes.append(shape)
        output_states.append(state)

    return context.Response(
        body=json.dumps({"shapes": output_shapes, "states": output_states}),
        headers={},
        content_type="application/json",
        status_code=200,
    )
