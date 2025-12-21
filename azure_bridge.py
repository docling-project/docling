from docling.document_converter import DocumentConverter
import os

print("🚀 Step 1: IBM Docling is starting up...")
converter = DocumentConverter()

# Using a real IBM research paper as our test subject
source = "https://arxiv.org/pdf/2408.09869.pdf" 
print(f"📄 Analyzing document: {source}")

result = converter.convert(source)
clean_markdown = result.document.export_to_markdown()

print("✅ SUCCESS! IBM Docling has parsed the data.")
print("\n--- DATA PREVIEW FOR LLOYD ---")
print(clean_markdown[:300] + "...")

# Save a local copy to show we have 'Azure-ready' data
with open("azure_ready_data.md", "w") as f:
    f.write(clean_markdown)
print("\n💾 Saved as: azure_ready_data.md")
