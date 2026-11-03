`formal` is {"mode": "holds" or "undetermined", "statements": [...]}. Each statement is an object {"if": [conditions], "then": result}, read "for every x, if all the conditions hold, then the result holds". For a plain statement, leave "if" empty: {"if": [], "then": "dog(rex)"}.

Conditions and results are literals:

- an atom: `dog(rex)`, `likes(ann, bob)`, `raining`
- a negation: `not dog(rex)`
- a comparison of numbers: `age(bob) >= 18`, `price(a) + 2 = price(b)`

A result can also be either/or: `cat(x) | dog(x)`. For example, "dogs that aren't small bark" is {"if": ["dog(x)", "not small(x)"], "then": "barks(x)"}.

`x`, `y`, `z`, `u`, `v` and `w` are variables: they stand for any individual. Every other name is a specific individual, property or relation. There's no "some" or "exists": name the individual instead. Names are lowercase, with underscores: `rex`, `has_fur`.
