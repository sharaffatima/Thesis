# Thesis Template

This is a LaTeX template for thesis projects of the Distributed Systems chair at the University of Duisburg-Essen.

## Usage

**Please read the [main file](thesis.tex) in its entirety.**
The options and packages used are explained there.
You can change it to better fit your needs.

_Also read the included example PDF
as it contains some additional information on usage
and resources for learning LaTeX._

Organize your chapters inside the _content_ folder and include the files in _thesis.tex_.
This makes it easier to keep track of smaller changes and single parts of your thesis.
Make sure to properly edit the [metadata](metadata.tex).  
Do _not_ edit the _clause.tex_ but remember to read and sign it before submitting your work.
Also remember to comment out the color changes made to references for printing.  
You can use the `\todo{}` command to include easily visible comments in the document.

## Requirements

The easiest way to ensure all required packages are installed is to utilize a preconfigured [Docker](https://docs.docker.com/get-docker/) container like [this one](https://github.com/dante-ev/docker-texlive).  
However, it should also work with locally installed tools and in various common LaTeX editors,
including [Overleaf](https://www.overleaf.com) and [TeXStudio](https://www.texstudio.org/).

### Manual

If [GNU make](https://www.gnu.org/software/make/) is available on your system, you can simply run `make setup` and `make` to compile your documents.
Make is commonly available on macOS and Linux, and it is possible to install on Windows: [further information](https://github.com/mbuilov/gnumake-windows)  
You can also just invoke the required commands directly.
To see these, just have a look at the [Makefile](Makefile).
Its structure is simple, the required commands are written in the lines starting with tabs.

### Automated

This template can also be used in combination with tools like [Visual Studio Code](https://code.visualstudio.com/) and the [LaTeX Workshop extension](https://github.com/James-Yu/LaTeX-Workshop).
It should work out of the box.  
If you wish to utilize the Docker container, all you need to do is add the following two options to your VS Code settings:
```json
    "latex-workshop.docker.enabled": true,
    "latex-workshop.docker.image.latex": "danteev/texlive",
```

## Removing the Glossary

If you don't want to use the glossary, you can comment the following lines from the `thesis.tex` file.
```latex
\chapter*{Glossary}
\printglossary[type=\acronymtype,title={List of Acronyms}]
\printglossary[title={Glossary}]
```

In this case you also need to change the `Makefile`, as `make pdf` also requires only one run of `pdflatex` and no execution of `makeglossaries`.


## Additional Information

It is useful to utilize a spell checking tool
like [LTex](https://marketplace.visualstudio.com/items?itemName=valentjn.vscode-ltex) or [cSpell](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker)
in order to catch the most blatant errors
that might occur when writing for prolonged periods.

### Usage of AI Tools

There is an explicit policy on the usage of AI based tools
for the creation of a thesis.
When signing the _Eidesstattliche Versicherung_ you state that you are aware of it
and followed it in your work.
Please read the policy on [our page](https://vs.uni-due.de/teaching/thesis/ai_tools_usage/).
