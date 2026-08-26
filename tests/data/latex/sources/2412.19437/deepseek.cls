\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{deepseek} % no versioning; versioning does not work well with arxiv.

%%======== Pckage options and base class ========
\RequirePackage{xifthen}
\newboolean{nonumbering}
\newboolean{logo}
\newboolean{address}
\newboolean{copyright}
\newboolean{internal}
\DeclareOption{nonumbering}{\setboolean{nonumbering}{false}}
\DeclareOption{logo}{\setboolean{logo}{true}}
\DeclareOption{address}{\setboolean{address}{true}}
\DeclareOption{copyright}{\setboolean{copyright}{true}}
\DeclareOption{internal}{\setboolean{internal}{false}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{extarticle}}
\ProcessOptions
\LoadClass{extarticle}

%%======== Packages Required ========
% Fonts and language
\AtEndOfClass{\RequirePackage[tracking=smallcaps]{microtype}}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel}
\RequirePackage{amsthm, amsmath, amsfonts, amssymb}
\RequirePackage{cmap}
\RequirePackage[T1]{fontenc}
% JH: switched to xurl from url due to wrapping issues. This fixes it. For example, the Megatron-Turing NLG link
\RequirePackage{xurl}
% \RequirePackage[colorlinks=true, allcolors=blue]{hyperref}  % cite is blue
% \RequirePackage[colorlinks=true, allcolors=black]{hyperref}
\RequirePackage{hyperref}
\RequirePackage{pifont}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{textcomp} % For copyright symbol styling
\RequirePackage[overload]{textcase}
%\RequirePackage{balance}
%% *Graphics, tables and other formatting
\RequirePackage{graphicx}
\RequirePackage{colortbl}
\RequirePackage{booktabs}
\RequirePackage{changepage}
\RequirePackage{enumitem} % For reducing bullet list item separation
\RequirePackage{tabularx}
\RequirePackage{datetime}
% Headers and footers
\RequirePackage{fancyhdr}  % custom headers/footers
\RequirePackage{lastpage}  % Number of pages in the document
% Lists and their spacing
\RequirePackage{enumitem} 
\setlist[itemize]{noitemsep}
\setlist[enumerate]{noitemsep}
% Title, Abstract, Bibliography
\RequirePackage[explicit]{titlesec}
\RequirePackage{bibentry}
\RequirePackage{mdframed}
\RequirePackage{caption}
\RequirePackage{needspace}

%%======== Can use any of these fonts ======== 
%\RequirePackage{charter}
%\RequirePackage[scaled]{helvet}
%\RequirePackage{bookman}
%\RequirePackage{times}
%\RequirePackage[default]{opensans} %fosfamily
%\usepackage{antpolt}
% Math fonts
\RequirePackage{mathpazo} % Palatino
%\usepackage{euler} % for math fints

% Use Bitstream Charter as the main font. Does not include a typewriter font.
% \RequirePackage{XCharter}
\RequirePackage[xcharter,bigdelims,vvarbb]{newtxmath}

% Use the typewriter font from lmodern, scaled up to match Charter's xheight.
% You may wish to add the "p" option to use the proportional font instead of
% the monospaced one, which might look nicer in cases where you do not need
% aligned columns
\RequirePackage[scaled=1.1]{zlmtt}

%%======== Margins and column spacing ========
\RequirePackage[left=2.5cm,%
right=2.5cm,%
top=3cm,%
bottom=3cm,%
headheight=40pt,%
headsep=20pt,%
%showframe, % for testing
a4paper]{geometry}%
\setlength{\columnsep}{15pt} % Distance between the two columns of text
%\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5\baselineskip}

%% *Custom font sizes and styles
\newcommand{\watermarkfont}{\color{black50}\normalfont\fontsize{10}{12}\selectfont}
\newcommand{\headerfont}{\normalfont\sffamily\fontsize{8}{10}\selectfont}
% \newcommand{\titlefont}{\color{black}\normalfont\bfseries\fontsize{18}{20}\selectfont} 
\newcommand{\titlefont}{\color{black}\normalfont\bfseries\fontsize{16}{20}\selectfont} 

\newcommand{\datesfont}{\normalfont\sffamily\itshape\fontsize{8}{10}\selectfont}
\newcommand\urlheaderfont{\color{blue}\fontsize{8}{10}\selectfont}
\newcommand{\absfont}{\linespread{1.2}\fontsize{11}{12}\selectfont}
\newcommand{\keywordstitlefont}{\bfseries\linespread{1.2}\fontsize{8}{10}\selectfont}
\newcommand{\keywordsfont}{\itshape\linespread{1.2}\fontsize{11}{12}\selectfont}
\newcommand{\copyrightfont}{\normalfont\fontsize{7}{9}\selectfont}
% \newcommand{\footerfont}{\color{black}\normalfont\fontsize{10}{10}\selectfont}
\newcommand{\footerfont}{\color{black}\normalfont\fontsize{11}{12}\selectfont}
\newcommand{\subjectfont}{\color{white}\bfseries\fontsize{10}{12}\selectfont}
\newcommand{\headingfont}{\color{black}\bfseries\fontsize{13}{14}\selectfont}

\newcommand{\keywords}[1]{\def\@keywords{#1}}
\newtoks\correspondingauthor
\newtoks\paperurl
\newtoks\reportnumber

% Dates and Times for footer
%\newdateformat{monthyeardate}{\THEDAY{ }\monthname[\THEMONTH] \THEYEAR}
% ISO-8601 date format
\newdateformat{monthyeardate}{\THEYEAR-\THEMONTH-\THEDAY}

\setcounter{secnumdepth}{3}
\ifthenelse{\boolean{nonumbering}}{\setcounter{secnumdepth}{0}}{}

%%========  Author and affiliation ========
\RequirePackage{authblk}
\setcounter{Maxaffil}{0}
% \setlength{\affilsep}{0.2em}  %
% \renewcommand\Authfont{\normalfont\bfseries\fontsize{11}{15}\selectfont\centering}
\renewcommand\Authfont{\normalfont\fontsize{11}{15}\selectfont\centering}
\renewcommand\Affilfont{\bfseries\fontsize{13}{15}\selectfont\centering}
\makeatletter
\renewcommand\AB@affilsepx{ , \protect\Affilfont}
\makeatother
% \renewcommand\Authands{ and }
\renewcommand\Authands{, }

\newcommand{\internalonly}{DeepSeek, Confidential. Internal Only.}
\newcommand{\copyrightext}{\footerfont \textcopyright\, \the\year{} DeepSeek. All rights 
reserved}


%%======== Header and Footer Content ========
%
% Header and Footer for the first page
% \pagestyle{fancy}   % Enables the custom headers/footers

\fancypagestyle{firststyle}{
    \fancyhead[R]{
        % Other content for the left side of the header, if any
    }

    \fancyhead[L]{
        \ifthenelse{\boolean{logo}}{
            \includegraphics[width=120pt]{logo/DeepSeeK}
        }{}
        % Other content for the right side of the header
        \ifdefined\paperurl
        \if\relax\the\paperurl\relax \else 
            \href{\the\paperurl}{\urlheaderfont \itshape \the\paperurl}\\ \fi
        \else \fi
        {\footerfont\itshape\monthyeardate\today}
    }
    % \fancyhead[C]{\textbf{Your Title Here}}
    \fancyhead[C]{}
    
    %
    \fancyfoot[L]{
    	\ifdefined\correspondingauthor
    	\if\relax\the\correspondingauthor\relax 
    	\else \footerfont {*\the\correspondingauthor \\} \fi
    	\else \fi
        \ifthenelse{\boolean{internal}}{\footerfont \internalonly \\}{\footerfont\bfseries\relax}
    	\ifthenelse{\boolean{address}}{
    		\itshape\footerfont DeepSeek, China \\}{} 
    	% \ifthenelse{\boolean{copyright}}{\copyrightext}{}
    }
    \fancyfoot[R]{
    	\ifthenelse{\boolean{internal}}{
    	\ifdefined\reportnumber
    	\if\relax\the\reportnumber\relax 
    	\else \footerfont\itshape  {\footerfont \bfseries
    % 	DeepSeek Technical Report \the\reportnumber
    	} \fi
    	\else \fi
    	}{\footerfont\bfseries\relax}
    }
    \fancyfoot[C]{\footerfont\bfseries\relax}
}

% Header and Footer for pages from 2 onwards
\fancyhead[L,R]{}
\makeatletter
\fancyhead[C]{\footerfont DeepSeek LLM: Scaling Open-Source Language Models with Longtermism}
% \fancyhead[C]{DeepSeek Technical Report}

\makeatother
\fancyfoot[C]{}
\fancyfoot[L]{
	\ifthenelse{\boolean{internal}}{\footerfont \internalonly\newline 
		\ifdefined\reportnumber
		\if\relax\the\reportnumber\relax 
		\else \footerfont {} \fi
		\else \fi}{\footerfont\bfseries\relax}
}
\fancyfoot[C]{\footerfont \thepage}

% Header and footer rules (coloured).
\renewcommand{\headrulewidth}{1pt}
\renewcommand{\footrulewidth}{1pt}
\let\DefaultHeadRule\headrule
\let\DefaultFootRule\footrule
\renewcommand{\headrule}{\color{gray}\DefaultHeadRule}
\renewcommand{\footrule}{\textcolor{gray}{\DefaultFootRule}}

\renewenvironment{abstract}
{\par\noindent\\  \ignorespaces \bfseries}
{\par\bigskip}


%%======== Abstract and section formatting ========
\def\xabstract{abstract}
\long\def\abstract#1\end#2{\def\two{#2}\ifx\two\xabstract 
	\long\gdef\theabstract{\ignorespaces#1}
	\def\go{\end{abstract}}\else
\typeout{^^J^^J PLEASE DO NOT USE ANY \string\begin\space \string\end^^J
	COMMANDS WITHIN ABSTRACT^^J^^J}#1\end{#2}
\gdef\theabstract{\vskip12pt BADLY FORMED ABSTRACT: PLEASE DO
NOT USE {\tt\string\begin...\string\end} COMMANDS WITHIN
THE ABSTRACT\vskip12pt}\let\go\relax\fi
\go}

% Define an environment with abstract content and styling
\makeatletter
\newcommand{\abscontent}{
	\noindent
        \centerline{\fontsize{15pt}{14pt}\selectfont\textbf{Abstract}}\vspace{5ex}
	\parbox{\dimexpr\linewidth}{\absfont \theabstract}
	\@ifundefined{@keywords}{}{
		\vskip1em \noindent \keywordsfont  Keywords: \@keywords}
}

\if@twocolumn%
\renewcommand{\maketitle}{
	\twocolumn[
	\begin{adjustwidth}{0pt}{0pt}
		\begin{flushleft}
			{
				{ \raggedright \titlefont 
				\@title\par}%
				\vskip11pt
				{\raggedright \@author\par}
				\vskip20pt%
			}%
		\end{flushleft}
	\end{adjustwidth}

		\abscontent
	\vskip20pt%
	]
	\thispagestyle{firststyle}
}%
\else% \@twocolumnfalse
\renewcommand{\maketitle}{\bgroup\setlength{\parindent}{0pt}
        \vspace*{3pt}
	\begin{adjustwidth}{0pt}{0pt}  
		\begin{flushleft}
			{
				{\raggedright \titlefont \@title\par}%
				\vskip20pt
				{\raggedright \@author\par}
				\vskip30pt %
			}%
		\end{flushleft}
	\end{adjustwidth}
	\egroup
	{%
		{\abscontent}
	}%
	\thispagestyle{firststyle}
}%
\fi
\makeatother

%%========  Bibliography using bibtex ========
\addto{\captionsenglish}{\renewcommand{\refname}{References}}
\newcommand{\citeas}[1]{\Needspace*{8\baselineskip}\noindent\bibentry{#1}}
% Uncomment the next line if you want the  numbered refs to be "1." instead of "[1]".
%\makeatletter \renewcommand\@biblabel[1]{#1.} \makeatother


\titleformat{\section}
{\large\bfseries\headingfont}
{\thesection.}
{0.5em}
%{\MakeUppercase{#1}}[]
{#1}
[]
\titleformat{name=\section,numberless}
{\large\bfseries\headingfont}
{}
{0em}
%{\MakeUppercase{#1}}[]
{#1}
[]
\titleformat{\subsection}
{\bfseries}
{\thesubsection.}
{0.5em}
{#1}
[]
\titleformat{\subsubsection}% [runin] If want to have inline for the subsubsect
{\bfseries\itshape}
{\thesubsubsection.}
{0.5em}
{#1}
[]

\titlespacing*{\section}{0pc}{3ex plus4pt minus3pt}{5pt}
%\titlespacing*{\section}{0pc}{1ex }{\baselineskip}
\titlespacing*{\subsection}{0pc}{2.5ex plus3pt minus2pt}{2pt}
\titlespacing*{\subsubsection}{0pc}{2ex plus2.5pt minus1.5pt}{2pt}
%\titlespacing*{\paragraph}{0pc}{2ex plus2pt minus1pt}{0pt}

%%======== Captions in tables and figures ========
\DeclareCaptionLabelSeparator{pipe}{ $\vert$ }% or $\vert$
\captionsetup{labelfont={}, %{bf,sf},%
labelsep=space,%
figurename=Figure,%
labelsep=pipe,
singlelinecheck=off,%
%justification=centering
}
