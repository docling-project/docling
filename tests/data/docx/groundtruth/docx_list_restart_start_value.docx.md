## List levels that restart at their start value

Each list has two levels. In both lists, level 2 starts at 3. When level 1 goes to the next number, level 2 starts again at 3. The text of each item is its number.

### Start value from w:start

The numbering definition of this list gives level 2 the start value 3 (w:start).

1. Item 1
    - 1.3. Item 1.3
    - 1.4. Item 1.4
2. Item 2
    - 2.3. Item 2.3
    - 2.4. Item 2.4
3. Item 3
    - 3.3. Item 3.3

### Start value from w:startOverride

The numbering definition of this list gives level 2 the start value 1 (w:start). The numbering instance of the list changes it to 3 (w:lvlOverride with w:startOverride).

1. Item 1
    - 1.3. Item 1.3
    - 1.4. Item 1.4
2. Item 2
    - 2.3. Item 2.3
    - 2.4. Item 2.4
3. Item 3
    - 3.3. Item 3.3