show: main.pdf
	zathura $<

main.pdf: main.md
	pandoc $< -o $@ --pdf-engine=xelatex -V geomtry:margin=1in
