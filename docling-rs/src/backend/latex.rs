use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;

use crate::models::common::{DocItemLabel, GroupLabel, InputFormat};
use crate::models::document::{create_doc_from_file, DoclingDocument};
use crate::models::table::TableCell;

use super::Backend;

struct CleanLatexPatterns {
    replacements: Vec<(Regex, &'static str)>,
}

fn clean_latex_patterns() -> &'static CleanLatexPatterns {
    static PATTERNS: OnceLock<CleanLatexPatterns> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        let defs: &[(&str, &str)] = &[
            (r"\\textbf\{([^}]*)\}", "$1"),
            (r"\\textit\{([^}]*)\}", "$1"),
            (r"\\emph\{([^}]*)\}", "$1"),
            (r"\\underline\{([^}]*)\}", "$1"),
            (r"\\texttt\{([^}]*)\}", "$1"),
            (r"\\text\{([^}]*)\}", "$1"),
            (r"\\mathrm\{([^}]*)\}", "$1"),
            (r"\\mathbf\{([^}]*)\}", "$1"),
            (r"\\href\{[^}]*\}\{([^}]*)\}", "$1"),
            (r"\\url\{([^}]*)\}", "$1"),
            (r"\\footnote\{[^}]*\}", ""),
            (r"\\label\{[^}]*\}", ""),
            (r"\\ref\{([^}]*)\}", "[$1]"),
            (r"\\eqref\{([^}]*)\}", "[$1]"),
            (r"\\cite\{([^}]*)\}", "[$1]"),
            (r"\\citep\{([^}]*)\}", "[$1]"),
            (r"\\citet\{([^}]*)\}", "[$1]"),
            (r"~", " "),
            (r"\\,", " "),
            (r"\\;", " "),
            (r"\\ ", " "),
            (r"\\&", "&"),
            (r"\\%", "%"),
            (r"\\#", "#"),
            (r"\\\$", "$"),
            (r"\\\\", ""),
        ];
        let replacements = defs
            .iter()
            .filter_map(|(pat, rep)| Regex::new(pat).ok().map(|re| (re, *rep)))
            .collect();
        CleanLatexPatterns { replacements }
    })
}

pub struct LatexBackend;

impl Backend for LatexBackend {
    fn convert(&self, path: &Path) -> anyhow::Result<DoclingDocument> {
        let mut doc = create_doc_from_file(path, &InputFormat::Latex)?;
        let content = std::fs::read_to_string(path)?;

        let base_dir = path.parent().unwrap_or(Path::new("."));
        let content = resolve_inputs(&content, base_dir, 0);

        parse_latex(&content, &mut doc);
        Ok(doc)
    }
}

fn resolve_inputs(content: &str, base_dir: &Path, depth: u32) -> String {
    if depth > 5 {
        return content.to_string();
    }
    let input_re = Regex::new(r"\\input\{([^}]+)\}").unwrap();
    let mut result = content.to_string();

    let captures: Vec<_> = input_re.captures_iter(content).collect();
    for cap in captures {
        let filename = &cap[1];
        let full_match = &cap[0];

        let mut file_path = base_dir.join(filename);
        if !file_path.exists() && file_path.extension().is_none() {
            file_path.set_extension("tex");
        }

        if let Ok(included) = std::fs::read_to_string(&file_path) {
            let resolved = resolve_inputs(&included, base_dir, depth + 1);
            result = result.replacen(full_match, &resolved, 1);
        }
    }
    result
}

fn parse_latex(content: &str, doc: &mut DoclingDocument) {
    let lines: Vec<&str> = content.lines().collect();
    let mut current_parent: Option<String> = None;
    let mut i = 0;

    let section_re =
        Regex::new(r"^\\(section|subsection|subsubsection|chapter|paragraph)\*?\{([^}]+)\}")
            .unwrap();
    let title_re = Regex::new(r"^\\title\{([^}]+)\}").unwrap();
    let begin_re = Regex::new(r"^\\begin\{(\w+\*?)\}").unwrap();
    let end_re = Regex::new(r"^\\end\{(\w+\*?)\}").unwrap();
    let item_re = Regex::new(r"^\\item\s*(.*)").unwrap();
    let inline_math_re = Regex::new(r"\$([^$]+)\$").unwrap();
    static CAPTION_RE: OnceLock<Regex> = OnceLock::new();
    let caption_re = CAPTION_RE.get_or_init(|| Regex::new(r"\\caption\{([^}]*)\}").unwrap());

    while i < lines.len() {
        let line = lines[i].trim();

        if line.is_empty() || line.starts_with('%') {
            i += 1;
            continue;
        }

        // Skip \title — not emitted in groundtruth
        if title_re.is_match(line) {
            i += 1;
            continue;
        }

        // Sections
        if let Some(caps) = section_re.captures(line) {
            let cmd = &caps[1];
            let text = clean_latex(&caps[2]);
            let level = match cmd {
                "chapter" => 1,
                "section" => 1,
                "subsection" => 2,
                "subsubsection" => 3,
                "paragraph" => 4,
                _ => 1,
            };
            if !text.is_empty() {
                let idx = doc.add_section_header(&text, level, None);
                current_parent = Some(format!("#/texts/{}", idx));
            }
            i += 1;
            continue;
        }

        // Display math: $$...$$
        if line.starts_with("$$") {
            if line.ends_with("$$") && line.len() > 4 {
                let formula = &line[2..line.len() - 2];
                if !formula.trim().is_empty() {
                    doc.add_text(
                        DocItemLabel::Formula,
                        formula.trim(),
                        current_parent.as_deref(),
                    );
                }
                i += 1;
            } else {
                i += 1;
                let mut formula = String::new();
                while i < lines.len() {
                    let l = lines[i].trim();
                    if l.starts_with("$$") || l.ends_with("$$") {
                        if l.len() > 2 && l.ends_with("$$") && !l.starts_with("$$") {
                            if !formula.is_empty() {
                                formula.push('\n');
                            }
                            formula.push_str(&l[..l.len() - 2]);
                        }
                        i += 1;
                        break;
                    }
                    if !formula.is_empty() {
                        formula.push('\n');
                    }
                    formula.push_str(l);
                    i += 1;
                }
                if !formula.is_empty() {
                    doc.add_text(
                        DocItemLabel::Formula,
                        formula.trim(),
                        current_parent.as_deref(),
                    );
                }
            }
            continue;
        }

        // Begin environment
        if let Some(caps) = begin_re.captures(line) {
            let env = caps[1].to_string();
            match env.as_str() {
                "itemize" | "enumerate" | "description" => {
                    let gidx = doc.add_group("list", GroupLabel::List, current_parent.as_deref());
                    let group_ref = format!("#/groups/{}", gidx);

                    i += 1;
                    while i < lines.len() {
                        let l = lines[i].trim();
                        if end_re.is_match(l) {
                            i += 1;
                            break;
                        }
                        if let Some(item_caps) = item_re.captures(l) {
                            let text = clean_latex(&item_caps[1]);
                            if !text.is_empty() {
                                doc.add_list_item(&text, false, Some(""), &group_ref);
                            }
                        }
                        i += 1;
                    }
                    continue;
                }
                "tabular" | "table" | "longtable" => {
                    let table_end = format!("\\end{{{}}}", env);
                    i += 1;
                    let mut rows: Vec<Vec<String>> = Vec::new();
                    let mut caption_text: Option<String> = None;
                    while i < lines.len() {
                        let l = lines[i].trim();
                        if l.starts_with(&table_end) {
                            i += 1;
                            break;
                        }
                        if l.starts_with("\\begin{tabular}") || l.starts_with("\\end{tabular}") {
                            i += 1;
                            continue;
                        }
                        if l == "\\hline"
                            || l == "\\toprule"
                            || l == "\\midrule"
                            || l == "\\bottomrule"
                        {
                            i += 1;
                            continue;
                        }
                        if l.starts_with("\\caption") {
                            if let Some(cc) = caption_re.captures(l) {
                                let ct = clean_latex(&cc[1]);
                                if !ct.is_empty() {
                                    caption_text = Some(ct);
                                }
                            }
                            i += 1;
                            continue;
                        }
                        if l.contains('&') || l.contains("\\\\") {
                            let row_text = l.trim_end_matches("\\\\").trim();
                            let cols: Vec<String> =
                                row_text.split('&').map(|s| clean_latex(s.trim())).collect();
                            rows.push(cols);
                        }
                        i += 1;
                    }

                    if !rows.is_empty() {
                        let num_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0) as u32;
                        let num_rows = rows.len() as u32;
                        let mut cells = Vec::new();
                        for (row_idx, row) in rows.iter().enumerate() {
                            for (col_idx, text) in row.iter().enumerate() {
                                cells.push(TableCell {
                                    row_span: 1,
                                    col_span: 1,
                                    start_row_offset_idx: row_idx as u32,
                                    end_row_offset_idx: (row_idx + 1) as u32,
                                    start_col_offset_idx: col_idx as u32,
                                    end_col_offset_idx: (col_idx + 1) as u32,
                                    text: text.clone(),
                                    column_header: false,
                                    row_header: false,
                                    row_section: false,
                                    fillable: false,
                                    formatted_text: None,
                                });
                            }
                        }
                        doc.add_table(cells, num_rows, num_cols, current_parent.as_deref());
                    }

                    if let Some(ct) = caption_text {
                        doc.add_text(DocItemLabel::Text, &ct, current_parent.as_deref());
                    }
                    continue;
                }
                "equation" | "equation*" | "align" | "align*" | "math" | "displaymath"
                | "gather" | "gather*" | "multline" | "multline*" | "eqnarray" | "eqnarray*"
                | "flalign" | "flalign*" => {
                    i += 1;
                    let mut formula = String::new();
                    while i < lines.len() {
                        let l = lines[i].trim();
                        if end_re.is_match(l) {
                            i += 1;
                            break;
                        }
                        if !l.is_empty() && !l.starts_with('%') {
                            if !formula.is_empty() {
                                formula.push('\n');
                            }
                            formula.push_str(l);
                        }
                        i += 1;
                    }
                    if !formula.is_empty() {
                        doc.add_text(DocItemLabel::Formula, &formula, current_parent.as_deref());
                    }
                    continue;
                }
                "verbatim" | "lstlisting" | "minted" => {
                    i += 1;
                    let mut code = String::new();
                    while i < lines.len() {
                        let l = lines[i];
                        if end_re.is_match(l.trim()) {
                            i += 1;
                            break;
                        }
                        if !code.is_empty() {
                            code.push('\n');
                        }
                        code.push_str(l);
                        i += 1;
                    }
                    if !code.is_empty() {
                        doc.add_text(DocItemLabel::Code, &code, current_parent.as_deref());
                    }
                    continue;
                }
                "abstract" => {
                    i += 1;
                    let mut abstract_text = String::new();
                    while i < lines.len() {
                        let l = lines[i].trim();
                        if end_re.is_match(l) {
                            i += 1;
                            break;
                        }
                        if !l.is_empty() && !l.starts_with('%') {
                            if !abstract_text.is_empty() {
                                abstract_text.push(' ');
                            }
                            abstract_text.push_str(&clean_latex(l));
                        }
                        i += 1;
                    }
                    if !abstract_text.is_empty() {
                        doc.add_text(
                            DocItemLabel::Text,
                            &abstract_text,
                            current_parent.as_deref(),
                        );
                    }
                    continue;
                }
                "figure" | "figure*" => {
                    i += 1;
                    while i < lines.len() {
                        if end_re.is_match(lines[i].trim()) {
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                    continue;
                }
                "document" => {
                    i += 1;
                    continue;
                }
                // Skip unknown environments to their matching \end
                _ => {
                    let target_end = format!("\\end{{{}}}", env);
                    i += 1;
                    while i < lines.len() {
                        let l = lines[i].trim();
                        if l.starts_with(&target_end) {
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                    continue;
                }
            }
        }

        // End environment (orphaned)
        if end_re.is_match(line) {
            i += 1;
            continue;
        }

        // Skip LaTeX preamble commands
        if line.starts_with('\\') {
            let cmd_end = line.find('{').unwrap_or(line.len());
            let cmd = &line[1..cmd_end];
            let skip_cmds = [
                "documentclass",
                "usepackage",
                "author",
                "date",
                "maketitle",
                "bibliography",
                "bibliographystyle",
                "label",
                "ref",
                "cite",
                "newcommand",
                "renewcommand",
                "def",
                "let",
                "setlength",
                "pagestyle",
                "thispagestyle",
                "tableofcontents",
                "listoffigures",
                "listoftables",
                "appendix",
                "newpage",
                "clearpage",
                "vspace",
                "hspace",
                "centering",
                "includegraphics",
                "caption",
                "footnote",
                "thanks",
                "email",
                "affiliation",
                "institute",
                "keywords",
                "DeclareMathOperator",
                "PassOptionsToPackage",
                "newblock",
                "bibitem",
                "shorttitle",
                "shortauthors",
                "ead",
                "color",
                "large",
                "And",
                "AND",
                "footnotemark",
            ];
            if skip_cmds.iter().any(|c| cmd.starts_with(c)) {
                i += 1;
                continue;
            }
        }

        // Regular text paragraph — also handles inline math extraction
        let text = clean_latex(line);
        if !text.is_empty() && !text.starts_with('\\') && !text.starts_with('{') && text.len() > 2 {
            let mut para = text;
            i += 1;
            while i < lines.len() {
                let l = lines[i].trim();
                if l.is_empty()
                    || l.starts_with('\\')
                    || l.starts_with('%')
                    || l.starts_with("$$")
                    || begin_re.is_match(l)
                    || section_re.is_match(l)
                {
                    break;
                }
                para.push(' ');
                para.push_str(&clean_latex(l));
                i += 1;
            }

            emit_text_with_inline_math(doc, &para, &inline_math_re, current_parent.as_deref());
            continue;
        }

        i += 1;
    }
}

fn emit_text_with_inline_math(
    doc: &mut DoclingDocument,
    text: &str,
    math_re: &Regex,
    parent: Option<&str>,
) {
    if !math_re.is_match(text) {
        doc.add_text(DocItemLabel::Text, text, parent);
        return;
    }

    let mut last_end = 0;
    for m in math_re.find_iter(text) {
        let before = text[last_end..m.start()].trim();
        if !before.is_empty() {
            doc.add_text(DocItemLabel::Text, before, parent);
        }
        doc.add_text(DocItemLabel::Text, m.as_str(), parent);
        last_end = m.end();
    }
    let after = text[last_end..].trim();
    if !after.is_empty() {
        doc.add_text(DocItemLabel::Text, after, parent);
    }
}

fn clean_latex(text: &str) -> String {
    let patterns = clean_latex_patterns();
    let mut result = text.to_string();
    for (re, rep) in &patterns.replacements {
        result = re.replace_all(&result, *rep).to_string();
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_latex_basic_formatting() {
        assert_eq!(clean_latex(r"\textbf{bold}"), "bold");
        assert_eq!(clean_latex(r"\textit{italic}"), "italic");
        assert_eq!(clean_latex(r"\emph{emphasized}"), "emphasized");
        assert_eq!(clean_latex(r"\texttt{mono}"), "mono");
        assert_eq!(clean_latex(r"\underline{under}"), "under");
    }

    #[test]
    fn test_clean_latex_references_preserved_as_brackets() {
        assert_eq!(clean_latex(r"\cite{smith2020}"), "[smith2020]");
        assert_eq!(clean_latex(r"\citep{jones2021}"), "[jones2021]");
        assert_eq!(clean_latex(r"\citet{doe2019}"), "[doe2019]");
        assert_eq!(clean_latex(r"\ref{sec:intro}"), "[sec:intro]");
        assert_eq!(clean_latex(r"\eqref{eq:1}"), "[eq:1]");
    }

    #[test]
    fn test_clean_latex_special_chars() {
        assert_eq!(clean_latex(r"\&"), "&");
        assert_eq!(clean_latex(r"\%"), "%");
        assert_eq!(clean_latex(r"\#"), "#");
        assert_eq!(clean_latex(r"\$"), "$");
        assert_eq!(clean_latex("word~word"), "word word");
    }

    #[test]
    fn test_clean_latex_strips_footnotes_and_labels() {
        assert_eq!(clean_latex(r"text\footnote{a note}more"), "textmore");
        assert_eq!(clean_latex(r"text\label{lbl}"), "text");
    }

    #[test]
    fn test_clean_latex_href_and_url() {
        assert_eq!(
            clean_latex(r"\href{http://example.com}{link text}"),
            "link text"
        );
        assert_eq!(
            clean_latex(r"\url{http://example.com}"),
            "http://example.com"
        );
    }

    #[test]
    fn test_clean_latex_combined() {
        let input = r"See \cite{smith2020} for more details. Also refer to Section \ref{sec:math}.";
        let expected = "See [smith2020] for more details. Also refer to Section [sec:math].";
        assert_eq!(clean_latex(input), expected);
    }

    #[test]
    fn test_resolve_inputs_no_inputs() {
        let content = "Hello world\nNo inputs here.";
        let result = resolve_inputs(content, Path::new("/nonexistent"), 0);
        assert_eq!(result, content);
    }

    #[test]
    fn test_resolve_inputs_depth_limit() {
        let content = r"\input{recursive}";
        let result = resolve_inputs(content, Path::new("/nonexistent"), 6);
        assert_eq!(result, content);
    }

    #[test]
    fn test_parse_sections() {
        let content = r"\begin{document}
\section{First}
Some text.
\subsection{Nested}
More text.
\subsubsection{Deep}
Deep text.
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.texts.len(), 6);
        assert_eq!(doc.texts[0].label, DocItemLabel::SectionHeader);
        assert_eq!(doc.texts[0].text, "First");
        assert_eq!(doc.texts[0].level, Some(1));

        assert_eq!(doc.texts[1].label, DocItemLabel::Text);
        assert_eq!(doc.texts[1].text, "Some text.");

        assert_eq!(doc.texts[2].label, DocItemLabel::SectionHeader);
        assert_eq!(doc.texts[2].text, "Nested");
        assert_eq!(doc.texts[2].level, Some(2));

        assert_eq!(doc.texts[4].label, DocItemLabel::SectionHeader);
        assert_eq!(doc.texts[4].text, "Deep");
        assert_eq!(doc.texts[4].level, Some(3));
    }

    #[test]
    fn test_parse_section_with_label_suffix() {
        let content = r"\begin{document}
\section{Results} \label{sec:results}
Content here.
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.texts[0].label, DocItemLabel::SectionHeader);
        assert_eq!(doc.texts[0].text, "Results");
    }

    #[test]
    fn test_parse_display_math_multiline() {
        let content = r"\begin{document}
$$
x^2 + y^2 = z^2
$$
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.texts.len(), 1);
        assert_eq!(doc.texts[0].label, DocItemLabel::Formula);
        assert_eq!(doc.texts[0].text, "x^2 + y^2 = z^2");
    }

    #[test]
    fn test_parse_display_math_single_line() {
        let content = r"\begin{document}
$$x^2 + y^2 = z^2$$
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.texts.len(), 1);
        assert_eq!(doc.texts[0].label, DocItemLabel::Formula);
        assert_eq!(doc.texts[0].text, "x^2 + y^2 = z^2");
    }

    #[test]
    fn test_parse_equation_star_environment() {
        let content = r"\begin{document}
\begin{equation*}
E = mc^2
\end{equation*}
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.texts.len(), 1);
        assert_eq!(doc.texts[0].label, DocItemLabel::Formula);
        assert_eq!(doc.texts[0].text, "E = mc^2");
    }

    #[test]
    fn test_parse_align_star_environment() {
        let content = r"\begin{document}
\begin{align*}
a &= b + c \\
d &= e + f
\end{align*}
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.texts.len(), 1);
        assert_eq!(doc.texts[0].label, DocItemLabel::Formula);
    }

    #[test]
    fn test_parse_inline_math_extraction() {
        let content = r"\begin{document}
Inline math: $E = mc^2$
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.texts.len(), 2);
        assert_eq!(doc.texts[0].text, "Inline math:");
        assert_eq!(doc.texts[1].text, "$E = mc^2$");
    }

    #[test]
    fn test_parse_table_with_caption() {
        let content = r"\begin{document}
\begin{table}[h]
\begin{tabular}{|c|c|}
\hline
A & B \\
\hline
1 & 2 \\
\hline
\end{tabular}
\caption{My table}
\end{table}
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.tables.len(), 1);
        assert_eq!(doc.texts.len(), 1);
        assert_eq!(doc.texts[0].text, "My table");
    }

    #[test]
    fn test_parse_table_column_header_false() {
        let content = r"\begin{document}
\begin{tabular}{|c|c|}
Name & Age \\
Alice & 25 \\
\end{tabular}
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.tables.len(), 1);
        for cell in &doc.tables[0].data.table_cells {
            assert!(!cell.column_header, "column_header should be false");
        }
    }

    #[test]
    fn test_parse_list_items_not_enumerated() {
        let content = r"\begin{document}
\begin{enumerate}
\item First
\item Second
\end{enumerate}
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.texts.len(), 2);
        for t in &doc.texts {
            assert_eq!(t.enumerated, Some(false));
            assert_eq!(t.marker, Some("".to_string()));
        }
    }

    #[test]
    fn test_parse_skips_title() {
        let content = r"\documentclass{article}
\title{My Title}
\begin{document}
\maketitle
\section{Intro}
Text here.
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.texts.len(), 2);
        assert_eq!(doc.texts[0].label, DocItemLabel::SectionHeader);
        assert_eq!(doc.texts[0].text, "Intro");
    }

    #[test]
    fn test_parse_unknown_env_skipped_entirely() {
        let content = r"\begin{document}
\begin{thebibliography}{10}
\bibitem{foo} Some reference.
\end{thebibliography}
\section{After}
Done.
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.texts.len(), 2);
        assert_eq!(doc.texts[0].text, "After");
        assert_eq!(doc.texts[1].text, "Done.");
    }

    #[test]
    fn test_parse_figure_star_skipped() {
        let content = r"\begin{document}
\begin{figure*}
\centering
\includegraphics{image.png}
\caption{A figure}
\end{figure*}
\section{Next}
\end{document}";
        let mut doc = DoclingDocument::new("test", "test.tex", "application/x-tex", 0);
        parse_latex(content, &mut doc);

        assert_eq!(doc.texts.len(), 1);
        assert_eq!(doc.texts[0].text, "Next");
    }
}
