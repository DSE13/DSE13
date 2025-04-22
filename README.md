# DSE13
Project Database for Group 13 of the 2025 Spring DSE

----

# Overleaf Project Style Guide: Design Synthesis Exercise (DSE)

## 1. Core Principles

*   **Consistency is Key:** Stick to these guidelines to ensure the final report (Project Plan, Baseline, Mid-Term, Final, etc.) is unified and professional.
*   **British English:** Use British spelling and grammar throughout (e.g., "centre", "analyse", "colour").
*   **Formal Tone:** Maintain a formal, objective, and precise scientific/engineering tone suitable for the DSE.
*   **Collaboration:**
    *   Use Overleaf comments for discussion.
    *   Use `\todo{Your message}` for actionable items directly in the text. Resolve these before submitting deliverables.
*   **Modularity:** Write content in the designated `.tex` files within the `mainmatter/` or `appendix/` folders. Keep the main `report.tex` file for structure and preamble only.

## 2. Document Structure & Referencing

*   **Sections:** Use standard LaTeX commands: `\chapter`, `\section`, `\subsection`. WE DO NOT USE `\subsubsection`.
*   **Labelling:** **Crucially, label everything you might refer to:** figures, tables, sections, equations.
    *   Place the `\label{...}` command *immediately after* the `\caption{}` (for figures/tables) or the sectioning command (for headings).
    *   Use a consistent naming convention prefix: `ch:`, `sec:`, `subsec:`, `fig:`, `tab:`, `eq:`. (e.g., `\label{fig:drone_concept}`)
*   **Cross-referencing:** **Always use `\autoref{label}`** to refer to sections, figures, tables, equations, etc. (e.g., "As shown in `\autoref{fig:drone_concept}`..."). It automatically adds "Figure", "Section", etc. and creates a clickable link in the PDF. *Do not write "Figure 1" manually.*

## 3. Writing Style & Formatting

*   **Paragraphs:** Paragraphs are separated by vertical space (due to the `parskip` package). **Do not indent** the first line of paragraphs manually.
*   **Emphasis:**
    *   Use `\textit{text}` for emphasis or defining terms.
    *   Use `\textbf{text}` sparingly for strong emphasis or specific labels (like totals in tables).
*   **Capitalisation:**
    *   Use **Title case** for all headings (Chapters, Sections, etc.). Example: `\section{Requirements and Analysis}`.
    *   Use **Title case** for figure and table captions. Example: `\caption{Distribution of Empty Drone Weight During the Flight}`.
    *   **Title case** means capitlizing all "important" words, meaning everything except things like "and", "or", "the", and so on.
*   **Acronyms:** Define acronyms on first use: "Work Breakdown Structure (WBS)". Use the acronym consistently afterwards. Refer to the Abbreviations list (Section 7) in the Project Guide.
*   **DSE Terminology:** Use the specific terminology defined in the Project Guide (e.g., DID, WBS, RDT). Refer to Appendix C (Glossary) for definitions.

## 4. Units (`siunitx` package)

*   **MANDATORY:** Use the `siunitx` package commands for all numbers with units.
    *   `\qty{value}{unit}` for numbers with units (e.g., `\qty{250}{\gram}`, `\qty{10}{\kilo\meter\squared}`).
    *   `\si{unit}` for units alone (e.g., "mass in `\si{\gram}`").
    *   `\deg` for degree symbols (e.g., `\qty{30}{\deg}`). Pre-defined in the preamble.
*   Check the preamble (`report.tex`) for any custom-defined units relevant to the project (`\DeclareSIUnit`).

## 5. Tables

*   **Environment:** Use the `tabularx` environment. This ensures that text in columns to wrap automatically to fit the page width (use the `X` column type for this). Tables will also automatically break across pages. To this end, define a first head (using `\endfirsthead`) which appears at the very top of the table, and a head for any subsequent parts of the table that may have been broken across pages (using `\endhead`). Similarly for `\endfoot` and `\endlastfoot`.
*   **Caption & Label:** The `\caption{Descriptive Caption.}` command goes **ABOVE** the tabularx environment. Follow immediately with `\label{tab:description}`.
*   **Numbers:** Use the `S` column type (from `siunitx`) for columns containing numerical data, especially decimals, to ensure proper alignment. When using an `S` column, table headers must be wrapped in `{}`, as in `{Table Header {[}kg{]}}`, for example. If using `S` columns, add the `[table-format=x.x]` specifier, where x.x refers to the expected amount of digits before and after the decimal point. So a number like 420.69 would be `[table-format=3.2]`.
*   **Styling:** Use `\hline` for horizontal lines. Use the defined colours (`headergray`, `sectiongray`) via `\rowcolor{colorname}` for header rows or section breaks within tables for consistency. Keep styling simple and clean.

*   **Table Example:**
  <pre>
\footnotesize
\renewcommand{\arraystretch}{1.2}
\begin{tabularx}{\textwidth}{|X|S[table-format=2.3]|S[table-format=2.3]|S[table-format=3.1]|}
\caption{ATR 72-HE Centre of Gravity Summary}\label{tab:cg_range_he}\\
\hline
\rowcolor{headergray}
\textbf{CG Item} & 
\multicolumn{1}{c|}{\makecell{\textbf{CG Location} \\{\scriptsize (Point 0) {[}m{]}}}} & 
\multicolumn{1}{c|}{\makecell{\textbf{CG Location} \\{\scriptsize  (LEMAC) {[}m{]}}}} & 
\multicolumn{1}{c|}{\makecell{\textbf{CG Location} \\{\scriptsize  (LEMAC) {[}\% MAC{]}}}}
\hline \endfirsthead

\caption{ATR 72-HE Centre of Gravity Summary -- Continued}\\
\hline
\rowcolor{headergray}
\textbf{CG Item} & 
\multicolumn{1}{c|}{\makecell{\textbf{CG Location} \\{\scriptsize  (Point 0) {[}m{]}}}} & 
\multicolumn{1}{c|}{\makecell{\textbf{CG Location} \\{\scriptsize  (LEMAC) {[}m{]}}}} & 
\multicolumn{1}{c|}{\makecell{\textbf{CG Location} \\{\scriptsize  (LEMAC) {[}\% MAC{]}}}}
\hline \endhead

\multicolumn{4}{r}{\textit{Continued on next page}} \\
\endfoot

\hline \endlastfoot

$\text{OEW}_\text{HE+Batt}$ & 12.51 & 1.269 & 55.1 \\ \hline
Passengers (56, avg) & 10.868 & -0.373 & -16.2 \\ \hline
Forward Cargo Hold             & 4.309  & -6.932 & -301.0 \\ \hline
Aft Cargo Hold                 & 21.859 & 10.618 & 461.0 \\ \hline
\rowcolor{sectiongray}
\multicolumn{4}{|c|}{Fuel Load}\\
Fuel                 & 12.277 & 1.093 & 47.5 \\\hline\addlinespace[6pt]\hline
\bfseries MTOW      & \bfseries 12.036 & \bfseries 0.795 & \bfseries 34.5 \\
\end{tabularx}</pre>

## 6. Figures

*   **Environment:** Use the `figure` floating environment with placement specifiers like `[htbp]`.
*   **Caption & Label:** The `\caption{Descriptive Caption.}` command goes **BELOW** the figure content (`\includegraphics`). Follow immediately with `\label{fig:description}`.
*   **Files:** Store all images in the `figures/` subfolder. Include them using `\includegraphics[width=0.8\textwidth]{figures/filename.png}` (adjust the `width` or use `height`/`scale` as needed).
*   **Format:** Prefer vector graphics (PDF, EPS) for diagrams and plots. Use high-resolution PNG/JPG for photos or raster images.
*   **Side-by-Side:** Use `minipage` environments within a `figure` environment to place figures or tables next to each other.

## 7. Mathematics

*   **Environments:** Use standard LaTeX math environments:
    *   `$inline math$`
    *   `\begin{equation}` environment for single numbered equations.
    *   `\begin{align}` environment for multi-line numbered equations (usually aligned at `=`).
    *   `\[ unnumbered display math \]` for unnumbered display equations.
*   **Referencing:** Label numbered equations you need to refer to using `\label{eq:description}` **inside** the math environment and reference using `\autoref{eq:description}`.
*   **Text in Math:** Use `\text{...}` from the `amsmath` package for normal text within math mode (e.g., units or short labels).

## 8. Citations and References

*   **Source File:** Add all bibliographic entries to the `report.bib` file. Use unique, consistent citation keys (e.g., `AuthorYear`, `ShortTitleYear`).
*   **In-Text Citations:** Use `\cite{key}`.
*   **Project Guide Rules (Strict! See Project Guide Sec 2.6 & Appendix D):**
    *   **Websites/URLs:** Do **NOT** put URLs in the `report.bib` file or the final reference list. Instead, cite websites using **footnotes** directly in the text where the information is mentioned.
    *   **Footnote Format:** Use `\footnote{URL: http://... [cited DD Month YYYY]}`. Include the URL and the date you last accessed it.
    *   **Reference List Format:** Follow the **exact** format specified in Appendix D of the Project Guide for books, articles, reports, etc. in your `report.bib` file.

---
