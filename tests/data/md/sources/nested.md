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
* changing symbol

A nested HTML list:

<ul>
    <li>First item</li>
    <li>Second item with subitems:
        <ul>
            <li>Subitem 1</li>
            <li>Subitem 2</li>
        </ul>
    </li>
    <li>Last list item</li>
</ul>

<!--
Table nesting apparently not yet suported by HTML backend:

<table>
  <tr>
    <td>Cell</td>
    <td>Nested Table
      <table>
        <tr>
          <td>Cell 1</td>
		  <>
        </tr>
        <tr>
          <td>Cell 2</td>
        </tr>
        <tr>
          <td>Cell 3</td>
        </tr>
        <tr>
          <td>Cell 4</td>
        </tr>
      </table>
    </td>
  </tr>
  <tr><td>additional row</td></tr>
</table>
-->
