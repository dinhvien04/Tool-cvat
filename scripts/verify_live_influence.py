import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from app.client import NineRouterClient
from app.feedback import FeedbackDatabase
from app.service import annotate_image

db = FeedbackDatabase()
img = Image.open('test.jpg')
client = NineRouterClient()
res = annotate_image(image_source=img, client=client, feedback_db=db, enable_feedback=True)
print(f"Model used: {res.model_used}")
print(f"Duration: {res.total_duration_seconds:.2f}s")
print(f"Visual examples used: {res.visual_examples_used}")
print(f"Text rules used: {res.text_rules_used}")
print(f"Shapes detected count: {len(res.shapes)}")
for s in res.shapes[:10]:
    print(f"  Shape: {s.get('label')} | {s.get('type')} | conf={s.get('confidence')}")
