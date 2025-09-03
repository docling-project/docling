
# Core Concepts

I was trying to understand how Docling works and found that the concepts page was basically empty - it just said "browse core concepts" without explaining anything. So I wrote this overview to help other users understand the system before diving into the code.

## System Architecture

### [Architecture Overview](architecture.md)
This explains how all the pieces fit together. Docling works by:
- **Document Converter** - The main entry point you use
- **Backends** - Different parsers for different file types
- **Pipelines** - The processing steps that run your documents through
- **Options** - How you customize what happens

### [Document Model](docling_document.md)
This is the key insight - all documents (PDFs, Word docs, images, etc.) get converted to the same internal format. So you can work with them the same way regardless of what they started as.

## Processing Features

### [Chunking](chunking.md)
When you have a huge document, you need to break it into smaller pieces. Docling can do this intelligently:
- **Text Chunking** - Just split by size
- **Semantic Chunking** - Split by meaning (keeps related content together)
- **Hybrid Approaches** - Best of both worlds

### [Serialization](serialization.md)
How to save your processed documents so you don't have to re-process them every time:
- **JSON Export** - Save everything (including confidence scores)
- **Format Conversion** - Export to Markdown, HTML, etc.
- **State Management** - Pick up where you left off

### [Confidence Scores](confidence_scores.md)
Docling tells you how sure it is about what it extracted:
- **OCR Confidence** - How sure it is about text from images
- **Layout Confidence** - How sure it is about document structure
- **Table Confidence** - How sure it is about table data

## Extensibility

### [Plugin System](plugins.md)
You can extend Docling to do new things:
- **Custom Backends** - Support for new file formats
- **Custom Pipelines** - Specialized processing workflows
- **Integration Points** - Hook into other systems

## Learning Path

I've organized this so you can learn step by step:

### Beginner Level
1. **Start Here** → [Architecture Overview](architecture.md) - Basic system understanding
2. **Core Concept** → [Document Model](docling_document.md) - How data is structured
3. **Basic Usage** → [Chunking](chunking.md) - Document processing fundamentals

### Intermediate Level
4. **Advanced Processing** → [Serialization](serialization.md) - Data persistence
5. **Quality Metrics** → [Confidence Scores](confidence_scores.md) - Reliability assessment
6. **Customization** → [Plugin System](plugins.md) - Extending functionality

### Advanced Level
7. **Deep Dive** → [Architecture](../concepts/architecture.md) - System internals
8. **Integration** → [Usage Guide](../usage/) - Practical applications
9. **API Reference** → [Reference](../reference/) - Technical details

## Key Principles

### **Unified Representation**
All document formats become the same thing internally. This is really powerful because you can write code that works with any document type.

### **Pipeline Architecture**
Processing happens in stages that you can customize, replace, or extend. Like a factory assembly line for documents.

### **Local-First Design**
Everything runs on your machine. No data sent to the cloud unless you explicitly want it.

### **Extensible Framework**
You can add new capabilities without changing the core system.

## What's Next?

Now that you understand the concepts, try:
- **[Usage Examples](../examples/)** - See concepts in action
- **[Installation Guide](../installation/)** - Get Docling running
- **[API Reference](../reference/)** - Technical details
- **[Integration Guides](../integrations/)** - Connect with other tools

## Need Help?

If something doesn't make sense:
- **GitHub Issues** - Report bugs or request features
- **Discussions** - Ask questions and share ideas
- **Documentation** - Explore other sections for more details
- **Examples** - See practical implementations of these concepts
<div class="grid" style="text-align: center">
    <div class="card">
        <img loading="lazy" alt="Docling architecture" src="../assets/docling_arch.png" width="75%" />
        <hr />
        Docling architecture outline
    </div>
</div>
