#!/bin/bash
echo "Packages installed in current Conda environment:"
for pkg in `cat requirement.txt | cut -d" " -f 1`; do conda list $pkg | grep "^$pkg "; done | tee _mypkg.txt
echo "Differences to original versions used:"
diff _mypkg.txt requirement.txt
rm _mypkg.txt
echo "Done!"
