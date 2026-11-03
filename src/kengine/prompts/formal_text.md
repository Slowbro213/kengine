`formal` is {"mode": "holds" or "undetermined", "statements": [...]}. Each statement is a string in this language:

- an atom: `dog(rex)`, `likes(ann, bob)`, `raining`
- a negation: `not dog(rex)`
- a comparison of numbers: `age(bob) >= 18`, `price(a) + 2 = price(b)`
- a rule, read "for every x, if ... then ...": `dog(x) & not small(x) -> barks(x)`
- either/or: `cat(tom) | dog(tom)`, also as a rule's result: `pet(x) -> cat(x) | dog(x)`

`x`, `y`, `z`, `u`, `v` and `w` are variables: they stand for any individual. Every other name is a specific individual, property or relation. There's no "some" or "exists": name the individual instead. Names are lowercase, with underscores: `rex`, `has_fur`.
