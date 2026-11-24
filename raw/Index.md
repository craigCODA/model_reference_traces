# Raw vocabulary index

The [SQLite pool](language.sqlite) holds 2,578,195 original entries and 5,487,036 links. A shared key collects exact stored spellings and their occurrence counts. Links run both ways between stored text and the words, phrases, characters, or tokenizer pieces it contains.

[Response links](Responses.md) add `{@[[...]]}` continuations and index ordered input-to-response combinations.

## Look up and combine entries

Run these from `research/model-reference-traces`.

```powershell
py -3 vocabulary_links.py links '[[hello]]'
py -3 vocabulary_links.py combine '[[want]]' '[[to]]' '[[go]]'
py -3 vocabulary_links.py combine '[[app]]' '[[le]]' --join=
py -3 vocabulary_links.py links '[[apple]]' --markdown raw/apple.md
```

`combine` assembles any number of links in the order supplied. It reports how many times the assembled text occurs in the pool and finds stored entries linked to every requested piece. Use `--limit -1` to return every related key.

## Example from the index

<code>an apple tree</code>

### Stored spellings

- <code>AN APPLE TREE</code> · sequence × 1

### Parts

- [[#^entry-116965|an]]
- [[#^entry-117477|an apple]]
- [[#^entry-144893|apple]]
- [[#^entry-144953|apple tree]]
- [[#^entry-1697845|tree]]

### Linked entries

<code>an</code> ^entry-116965

<code>an apple</code> ^entry-117477

<code>apple</code> ^entry-144893

<code>apple tree</code> ^entry-144953

<code>tree</code> ^entry-1697845
