## Test with tables

A uniform table

| Header 0.0   | Header 0.1   | Header 0.2   |
|--------------|--------------|--------------|
| Cell 1.0     | Cell 1.1     | Cell 1.2     |
| Cell 2.0     | Cell 2.1     | Cell 2.2     |

A non-uniform table with horizontal spans

| Header 0.0   | Header 0.1          | Header 0.2          |
|--------------|---------------------|---------------------|
| Cell 1.0     | Merged Cell 1.1 1.2 | Merged Cell 1.1 1.2 |
| Cell 2.0     | Merged Cell 2.1 2.2 | Merged Cell 2.1 2.2 |

A non-uniform table with horizontal spans in inner columns

| Header 0.0   | Header 0.1          | Header 0.2          | Header 0.3   |
|--------------|---------------------|---------------------|--------------|
| Cell 1.0     | Merged Cell 1.1 1.2 | Merged Cell 1.1 1.2 | Cell 1.3     |
| Cell 2.0     | Merged Cell 2.1 2.2 | Merged Cell 2.1 2.2 | Cell 2.3     |

A non-uniform table with vertical spans

| Header 0.0   | Header 0.1          | Header 0.2   |
|--------------|---------------------|--------------|
| Cell 1.0     | Merged Cell 1.1 2.1 | Cell 1.2     |
| Cell 2.0     | Merged Cell 1.1 2.1 | Cell 2.2     |
| Cell 3.0     | Merged Cell 3.1 4.1 | Cell 3.2     |
| Cell 4.0     | Merged Cell 3.1 4.1 | Cell 4.2     |

A non-uniform table with all kinds of spans and empty cells

| Header 0.0   | Header 0.1          | Header 0.2   |    |                     |
|--------------|---------------------|--------------|----|---------------------|
| Cell 1.0     | Merged Cell 1.1 2.1 | Cell 1.2     |    |                     |
| Cell 2.0     | Merged Cell 1.1 2.1 | Cell 2.2     |    |                     |
| Cell 3.0     | Merged Cell 3.1 4.1 | Cell 3.2     |    |                     |
| Cell 4.0     | Merged Cell 3.1 4.1 | Cell 4.2     |    | Merged Cell 4.4 5.4 |
|              |                     |              |    | Merged Cell 4.4 5.4 |
|              |                     |              |    |                     |
|              |                     |              |    |                     |
|              |                     |              |    | Cell 8.4            |