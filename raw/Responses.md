# Response links

`[[hello]] {@[[hello]]}` is a response link. The braces hold one or more ordered links from the [raw vocabulary](Index.md). The index traces this example to the stored phrase `HELLO HELLO`.

The response index contains 1,718,057 distinct continuation pairs backed by 2,376,696 links to stored phrases. A lookup uses the longest ending phrase with a matching continuation and shows the exact stored spelling of an example.

## Look up or compose a response

Run these from `research/model-reference-traces`.

```powershell
py -3 response_links.py respond '[[hello]]'
py -3 response_links.py respond '[[hello]]' --reply '{@[[hello]]}'
py -3 response_links.py respond '[[hello]]' --length 2 --offset 278642537400 --limit 1
```

Use `--reply` to inspect a specific pairing, including multiple links such as `{@[[to]] [[go]]}`. Use `--length` to page through every ordered response of that many keys. Positions start at zero and repeated keys are allowed.

## Count the combinations

Each slot can use any of the 2,298,793 shared keys. A single input key paired with a single response key gives 5,284,449,256,849 possible pairs. With `i` input slots and `r` response slots, every input has `N^r` possible responses and the full index addresses `N^(i + r)` ordered pairs. The index computes each requested page directly from its positions.

Refresh the stored continuations with `py -3 response_links.py build` after rebuilding the vocabulary pool.
