<!-- Prepended (prose only, no heading) to the generated modules/medpc.md page. -->
Imports a MED-PC data file (`.txt`) and its program (`.MPC`) as an event stream. It reads one
program layout: event codes in array `E`, their times in seconds in array `T`, and an
`Event stamps ( array E )` block in the program naming each code. A file in another layout is
refused rather than guessed at, and a time past the session length `H` (the sign of a program
that logs time in ticks) is refused too.

MED-PC can save every box of a day into one file. A recording is one session, so a file holding
several is refused until the parameters below pick one; the refusal lists every session in the
file. Leave them unset for a file that holds one session.
