# Nesting

A list featuring nesting:

- abc
    - abc123
        - abc1234
            - abc12345
                - a.
                - b.
        - abcd1234：
            - abcd12345：
                - a.
                - b.
- def：
    - def1234：
        - def12345。
- after one empty line
    - foo
- afer two empty lines
    - bar

- changing symbol

A nested HTML list:

&lt;ul&gt;
    &lt;li&gt;First item&lt;/li&gt;
    &lt;li&gt;Second item with subitems:
        &lt;ul&gt;
            &lt;li&gt;Subitem 1&lt;/li&gt;
            &lt;li&gt;Subitem 2&lt;/li&gt;
        &lt;/ul&gt;
    &lt;/li&gt;
    &lt;li&gt;Last list item&lt;/li&gt;
&lt;/ul&gt;
&lt;!--
Table nesting apparently not yet suported by HTML backend:

&lt;table&gt;
  &lt;tr&gt;
    &lt;td&gt;Cell&lt;/td&gt;
    &lt;td&gt;Nested Table
      &lt;table&gt;
        &lt;tr&gt;
          &lt;td&gt;Cell 1&lt;/td&gt;
		  &lt;&gt;
        &lt;/tr&gt;
        &lt;tr&gt;
          &lt;td&gt;Cell 2&lt;/td&gt;
        &lt;/tr&gt;
        &lt;tr&gt;
          &lt;td&gt;Cell 3&lt;/td&gt;
        &lt;/tr&gt;
        &lt;tr&gt;
          &lt;td&gt;Cell 4&lt;/td&gt;
        &lt;/tr&gt;
      &lt;/table&gt;
    &lt;/td&gt;
  &lt;/tr&gt;
  &lt;tr&gt;&lt;td&gt;additional row&lt;/td&gt;&lt;/tr&gt;
&lt;/table&gt;
--&gt;
