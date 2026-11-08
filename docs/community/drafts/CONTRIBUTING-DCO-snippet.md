## Sign your commits (DCO)

By contributing you certify the [Developer Certificate of Origin 1.1](https://developercertificate.org/): the work is yours or you may submit it under the project licence.
Add a sign-off to every commit:

```
git commit -s -m "fix(ptx): ..."     # appends: Signed-off-by: Your Name <your-email>
git rebase --signoff origin/integration   # add it to existing commits
```

The sign-off must use your real name and a reachable address (a GitHub noreply address is fine).
Do not submit: NVIDIA headers, binaries or SASS disassembly; code copied from projects with an incompatible licence; third-party code without its licence notice.
