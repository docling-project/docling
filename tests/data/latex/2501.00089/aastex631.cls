%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  AASTEX631.cls                                 %%
%%  Jan 29f, 2021                                  %% 
%%                                                %%
%% Copyright 2021 American Astronomical Society   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\currversion{6.3.1f}

%%
%%    These files are distributed
%%    WITHOUT ANY WARRANTY; without even the implied warranty of
%%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%% 
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}%
%%
%%%     date            = "9/19/2016",
%%%     author          = "Amy Hendrickson (mailto:amyh@texnology.com)" 
%%%                            including 
%%%                       1) code from emulateapj.cls, version 01/23/15(marked below)
%%%                            written by Alexey Vikhlinin, which relies on RevTeX4-1
%%%                            for much functionality;
%%%                       2) some code from aastex (marked below)
%%%                       3) Original work by Amy Hendrickson (marked below) for
%%%                       extensive table tools, including easy decimal entry and splitting
%%%                       tables horizontally into 2 or 3 pieces, for tabular and deluxetable
%%%                       environments; rotate tables; gridlines for use in positioning figures, 
%%%                       track changes.

%%% This work may be distributed and/or modified under the 
%%% conditions of the LaTeX Project Public License, either version 1.3c 
%%% of this license or (at your option) any later version.
%%% The latest version of this license is in
%%%  http://www.latex-project.org/lppl.txt
%%% and version 1.3 or later is part of all distributions of LaTeX 
%%% version 2005/12/01 or later.
%
%%% This work has the LPPL maintenance status `maintained'.
%
%%% The Current Maintainer of this work is Greg Schwarz <greg.schwarz@aas.org> 
%%% This work consists of the file aastex.cls

%%%     copyright       = "Copyright (C) 2016--2020 American Astronomical Society
%%%
%%%                        This work may be distributed and/or modified under the
%%%                        conditions of the LaTeX Project Public License, either version 1.3
%%%                        of this license or (at your option) any later version.
%%%                        The latest version of this license is in
%%%                        http://www.latex-project.org/lppl.txt
%%%                        and version 1.3 or later is part of all distributions of LaTeX
%%%                        version 2003/12/01 or later.
%%%
%%%                        This work has the LPPL maintenance status "maintained".
%%%
%%%                        The Current Maintainer of this work is the American Astronomical Society.
%%%
%%%                        This work consists of all files listed in the document README.
%%%
%%%     address         = "American Astronomical Society
%%%                        USA",
%%%     telephone       = "+1 ???",
%%%     FAX             = "",
%%%     email           = "aastex-help@aas.org",
%%%     codetable       = "ISO/ASCII",
%%%     keywords        = "latex, AAS, journal",
%%%     supported       = "yes",
%%%     abstract        = "formatter for AAS journal submissions",
%%%  }


\ProvidesClass{aastex631}%%%
 [2020/12/20 Version 6.3.1d/AAS markup document class]%
{}
\ClassInfo{aastex}{%
^^J
^^J
 Original \protect\LaTeX2.09 style
 by Chris Biemesderfer (chris@seagoat.com).
^^J
 Adapted to \protect\LaTeXe\space
 by A. Ogawa (ogawa@teleport.com)%
^^J
emulateapj.cls included, Copyright 2000-2015 Alexey Vikhlinin
^^J
Rewrite and update of emulateapj.cls,
revised and enhanced with table macros, gridlines, draft watermark,^^J
track changes, new `modern' style and much more, by Amy Hendrickson,%
^^J
 (amyh@texnology.com, http://www.texnology.com)
^^J
^^J
}%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% aastex.cls version 2.0 is a based on emulateapj.cls, version January 23, 2015,
%% and an older version of aastex.cls, with many additional functionalities.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Options. Document style options, and additional feature options.

% These are the General document style options, 6 choices

%   manuscript	: onecolumn, doublespace, 12pt fonts

%%  Preprints are like manuscript, except they are single spaced.
%   preprint	: onecolumn, single space, 12pt fonts
%   preprint2	: twocolumn, single space, 12pt fonts

%% Modern is new design developed by David Hogg and Daniel Foreman-Mackey
%   modern      : one column, single space, 12pt fonts, more stylish

%   twocolumn	: a two column article, single space, 10pt  fonts

% Default document style option
%   onecolumn	: a one column article; single space, 10pt fonts

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Additional feature options:

%   revtex4 	: will produce an error message and demand to download revtex4-1. 
%   twocolappendix: make 2 column appendix // produces error message. 
%      onecolappendix: make 1 column appendix is the default. // not needed because it is the default
%      astrosymb	: Loads Astrosymb font and define \astro commands. 
%      tighten	: Makes baselineskip slightly smaller, only works with twocolumn option.
%%    author can also use the commands \tighten, or \singlespace or \doublespace in the body of their article
%%    to get this change to the baselineskip
%      times	: uses times font as the default
%      linenumbers	: turn on lineno package.
%      trackchanges : required to see the revision mark up and print output
%      letteredappendix: (was numbered appendix) Not needed because it is the default, will label appendix sections A, B, ...
%      numberedappendix: Legacy command, will produce same results as lettered appendix, since we don't want more
%                        than one `section 1' in article: will cause problems for cross referencing.

%    anonymous: to not list authors/affiliations be listed

\newif\ifmanu
\newif\iftwelvepoint
\newif\ifmodern
\newif\ifdoublespace
\newif\if@two@col
\newif\ifonecolstyle
\newif\iftwocolstyle

\newif\ifrnaas
%% makes modern the style for rnaas
\DeclareOption{RNAAS}{\global\rnaastrue}
\DeclareOption{rnaas}{\global\rnaastrue}

 \DeclareOption{manuscript}{\twelvepointtrue\@two@colfalse\doublespacetrue\manutrue}%

%% Preprints are like manuscript but are single spaced:
\newif\ifpreprint
 \DeclareOption{preprint}{\@two@colfalse\preprinttrue\twelvepointtrue}%

\newif\ifpreprinttwo
 \DeclareOption{preprint2}{\@two@coltrue\preprinttwotrue\twelvepointtrue}% 


%% New design suggested by
\DeclareOption{modern}{\@two@colfalse\twelvepointtrue\moderntrue}

%% left in from older version, in case it could be useful:



\DeclareOption{twocolumn}{\onecolstylefalse\twocolstyletrue\@two@coltrue\twelvepointfalse}

%% default style
\DeclareOption{onecolumn}{\onecolstyletrue\twocolstylefalse}

%% obsolete option
\DeclareOption{revtex4}{\typeout{=================^^J^^J!!! The revtex4 option no longer
 allowed!^^J^^J Please download revtex4-1.cls^^J^^J
 Thank-you!^^J^^J=================}\stop}

\newif\if@two@col@app
\DeclareOption{twocolappendix}{\@two@col@apptrue}
\DeclareOption{onecolappendix}{\@two@col@appfalse}%% this is default

\newif\iflongauthor
\DeclareOption{longauthor}{\global\longauthortrue}

\newlength{\bibbaselineskip}\setlength{\bibbaselineskip}{3.075mm}

 \DeclareOption{astrosymb}{
   \def\astro#1{\leavevmode\hbox{\astro@font#1}}%
   \def\load@astro@font{%
\iftwelvepoint
 \font\astro@font=astrosym at 8pt
\else
 \font\astro@font=astrosym at 7pt
\fi
}%
   \AtBeginDocument{\load@astro@font}
 }

%% Default definition for \astro, in case astrosymb option has not
%% been used:
\def\astro#1{\typeout{^^J^^J Please use class option `astrosymb' to
access fonts for the \string\astro\space command
^^J^^J
}\stop}

   \def\apjsecfont{\normalsize}
   \def\secnum@size{\small}
   \def\subsecnum@size{\normalsize}

    \def\AppendixApjSectionMarkInTitle#1{\relax}
\newif\ifletteredappendix
     \DeclareOption{letteredappendix}{\global\letteredappendixtrue
       \def\AppendixApjSectionMarkInTitle#1{#1.\ }
     }

%% this option is obsolete, since letteredappendix is a more logical name;
%%  but keeping it here in case someone uses it, set to be the same as
%% letteredappendix.
     \DeclareOption{numberedappendix}{\global\letteredappendixtrue
       \def\AppendixApjSectionMarkInTitle#1{#1.\ }
     }

     \newif\if@number@appendix@floats\@number@appendix@floatsfalse
     \DeclareOption{appendixfloats}{
       \@number@appendix@floatstrue
     }

%%%%%%%%%%%%%
%% AH

%% This option changes the baselineskips in the definitions of \normalsize etc
%% to make them a bit smaller.
\newif\iftighten
\DeclareOption{tighten}{\global\tightentrue}

%% AH
\newif\iftimes
\DeclareOption{times}{\global\timestrue}

\newif\ifnumlines
\DeclareOption{linenumbers}{\global\numlinestrue}

\newif\iftrack
\DeclareOption{trackchanges}{\global\tracktrue}

\newif\ifanonymous
\DeclareOption{anonymous}{\global\anonymoustrue}


%% End AH Options

\ExecuteOptions{onecolumn,onecolappendix,letteredappendix}% 

\ProcessOptions	

%%%%%%%%%%%%%
%

 \iftimes
%% Might be better to use apjfonts, from Alexey Vikhlinin's website
%% http://hea-www.harvard.edu/~alexey/emulateapj/apjfonts.sty
%%\usepackage{apjfonts}
%% but times.sty is ubiquitous, and many people may not have apjfonts.sty
%% or the fonts that it calls.
%% default fonts:
\usepackage{times}
 \fi

%%%%%%%%%%%%%

 \IfFileExists{revtex4-1.cls}{
    \def\@revtex@cls{revtex4-1}
  }{
\typeout{^^J^^J Please update your system to include
revtex4-1.cls^^J^^J}\stop
  }
\def\@revtex@cls{revtex4-1}
\let\@startsection@from@latex=\@startsection


%% Passing options to revtex while loading:
  \iftwelvepoint
      \iflongauthor
         \LoadClass[nofootinbib,showkeys,twoside,floatfix,unsortedaddress,flushbottom,12pt,aps,pra]
          {\@revtex@cls}
     \else
        \LoadClass[nofootinbib,showkeys,twoside,floatfix,superscriptaddress,flushbottom,12pt,aps,pra]
       {\@revtex@cls}
     \fi
  \else %not twelvepoint
     \iflongauthor
         \LoadClass[nofootinbib,showkeys,twoside,floatfix,unsortedaddress,flushbottom,10pt,aps,pra]
         {\@revtex@cls}
     \else
        \LoadClass[nofootinbib,showkeys,twoside,floatfix,superscriptaddress,flushbottom,10pt,aps,pra]
        {\@revtex@cls}
    \fi
  \fi

%%%%%%%%%%%%%%%%%%%%

\def\blankaffiliation{~}%

%% needed?
 \def\@eapj@cap@font{\bfseries}
 \def\@eapj@figname{Figure}
 \def\@eapj@tabname{Table}

% Restore natbib package without sorting. 
\def\NAT@sort{0}
\def\NAT@cmprs{0}

% only redefine sort@cites for natbib versions < 2009
\@ifundefined{NAT@sort@cites@}{\def\NAT@sort@cites#1{\edef\NAT@cite@list{#1}}}%

\RequirePackage{latexsym}% 
\RequirePackage{graphicx}% 
\RequirePackage{amssymb}% 

\newcommand{\bibstyle@aas}{\bibpunct{(}{)}{;}{a}{}{,}}% 
\@citestyle{aas}% 

\let\@startsection=\@startsection@from@latex
\def\baselinestretch{1.0}

%% Default vertical space between horizontal lines. Can change definition
%% of arraystretch in the body of the paper; either for full article or
%% for a single table.
\AtBeginDocument{\def\arraystretch{1}}

%% needed?
\def\raggedcolumn@sw#1#2{#2}  % implement flushbottom as it was in revtex4

\RequirePackage{epsf,graphicx}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Font families
%% modification of bk10.clo, and bk12.clo, standard LaTeX distribution
%%
%% Implements doublespace used in `manuscript' option
%% Implements `tighten' option: make baselineskip smaller if `tighten' is used
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% tighten only works with two column styles:
\iftwocolstyle\else\global\tightenfalse\fi

%%% preprint, preprint2, manuscript are 12pt
%% Modified bk12.clo
\iftwelvepoint

\renewcommand\normalsize{%
\ifdoublespace % manuscript
  \iftighten
   \@setfontsize\normalsize\@xiipt{20}%
  \else
   \@setfontsize\normalsize\@xiipt{24}%
  \fi
\else
  \iftighten
   \@setfontsize\normalsize\@xiipt{13}%
   \else
      \ifmodern
      \@setfontsize\normalsize\@xiipt{16}%
      \else
        \@setfontsize\normalsize\@xiipt{15}%
      \fi
   \fi
\fi %% end doublespace/no double space
   \abovedisplayskip 12\p@ \@plus3\p@ \@minus7\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}

\normalsize

\renewcommand\small{%
\ifdoublespace % manuscript
   \@setfontsize\small\@xipt{20}%
\else
\ifmodern
   \@setfontsize\small\@xipt{12}%
\else
   \@setfontsize\small\@xipt{13.6}%
\fi\fi
%% end doublespace/no double space
   \abovedisplayskip 11\p@ \@plus3\p@ \@minus6\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 9\p@ \@plus3\p@ \@minus5\p@
               \parsep 4.5\p@ \@plus2\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}

\renewcommand\footnotesize{%
\ifdoublespace % manuscriptstyle
   \@setfontsize\footnotesize\@xpt{18}
\else
   \ifmodern
   \@setfontsize\footnotesize\@xpt{11}%
   \else
   \@setfontsize\footnotesize\@xpt\@xiipt
\fi\fi
   \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 6\p@ \@plus2\p@ \@minus2\p@
               \parsep 3\p@ \@plus2\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\ifdoublespace % manuscript style
\renewcommand\scriptsize{\@setfontsize\scriptsize\@viiipt{14}}
\renewcommand\tiny{\@setfontsize\tiny\@vipt{10}}
\renewcommand\large{\@setfontsize\large\@xivpt{18}}
\renewcommand\Large{\@setfontsize\Large\@xviipt{22}}
\renewcommand\LARGE{\@setfontsize\LARGE\@xxpt{25}}
\renewcommand\huge{\@setfontsize\huge\@xxvpt{30}}
\else
\renewcommand\scriptsize{\@setfontsize\scriptsize\@viiipt{9.5}}
\renewcommand\tiny{\@setfontsize\tiny\@vipt\@viipt}
\renewcommand\large{\@setfontsize\large\@xivpt{22}}
\renewcommand\Large{\@setfontsize\Large\@xviipt{24}}
\renewcommand\LARGE{\@setfontsize\LARGE\@xxpt{28}}
\renewcommand\huge{\@setfontsize\huge\@xxvpt{30}}
\fi
\let\Huge=\huge
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\else %% not twelvepoint, now tenpoint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% no doublespace here, because the only doublespace is in `manuscript style' which is 12pt

\renewcommand\normalsize{%
\iftighten
   \@setfontsize\normalsize\@xpt{11.5}%
\else
   \@setfontsize\normalsize\@xpt{12.5}%
\fi
   \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}

\normalsize

\renewcommand\small{%
\iftighten
   \@setfontsize\small\@ixpt{12}%
\else
   \@setfontsize\small\@ixpt{11}%
\fi
   \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 4\p@ \@plus2\p@ \@minus2\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}

\renewcommand\footnotesize{%
\iftighten
   \@setfontsize\footnotesize\@viiipt{9}%
\else
   \@setfontsize\footnotesize\@viiipt{9.5}% standard eightpoint size
\fi
   \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus\p@
   \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 3\p@ \@plus\p@ \@minus\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}

%% Standard 10pt font sizes
\renewcommand\scriptsize{\@setfontsize\scriptsize\@viipt\@viiipt}
\renewcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
\renewcommand\large{\@setfontsize\large\@xiipt{14}}
\renewcommand\Large{\@setfontsize\Large\@xivpt{18}}
\renewcommand\LARGE{\@setfontsize\LARGE\@xviipt{22}}
\renewcommand\huge{\@setfontsize\huge\@xxpt{25}}
\renewcommand\Huge{\@setfontsize\Huge\@xxvpt{30}}

\fi

%% End font sizes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}
\setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}
\setlength\headheight{12\p@}
\def\@listi{\leftmargin\leftmargini
            \parsep 4\p@ \@plus2\p@ \@minus\p@
            \topsep 8\p@ \@plus2\p@ \@minus4\p@
            \itemsep4\p@ \@plus2\p@ \@minus\p@}
\let\@listI\@listi
\@listi
\def\@listii {\leftmargin\leftmarginii
              \labelwidth\leftmarginii
              \advance\labelwidth-\labelsep
              \topsep    4\p@ \@plus2\p@ \@minus\p@
              \parsep    2\p@ \@plus\p@  \@minus\p@
              \itemsep   \parsep}
\def\@listiii{\leftmargin\leftmarginiii
              \labelwidth\leftmarginiii
              \advance\labelwidth-\labelsep
              \topsep    2\p@ \@plus\p@\@minus\p@
              \parsep    \z@
              \partopsep \p@ \@plus\z@ \@minus\p@
              \itemsep   \topsep}
\def\@listiv {\leftmargin\leftmarginiv
              \labelwidth\leftmarginiv
              \advance\labelwidth-\labelsep}
\def\@listv  {\leftmargin\leftmarginv
              \labelwidth\leftmarginv
              \advance\labelwidth-\labelsep}
\def\@listvi {\leftmargin\leftmarginvi
              \labelwidth\leftmarginvi
              \advance\labelwidth-\labelsep}

%  ****************************************************
%  *             TWO AND SINGLE ONE COLUMN STYLES: AV *
%  ****************************************************
%

%%%%% include atbeginend.sty by AV:

% Copy \begin and \end to \BeginEnvironment and \EndEnvironment
\let\BeginEnvironment=\begin
\let\EndEnvironment=\end
%% \ifundefined from TeXbook
\def\IfUnDef#1{\expandafter\ifx\csname#1\endcsname\relax}
% Null command needed to eat its argument
\def\NullCom#1{}
\def\begin#1{%
% if defined \BeforeBeg for this environment, execute it
\IfUnDef{BeforeBeg#1}\else\csname BeforeBeg#1\endcsname\fi%
\IfUnDef{AfterBeg#1}% This is done to skip the command for environments
                     % which can take arguments, like multicols; YOU MUST NOT
                     % USE \AfterBegin{...}{...} for such environments! 
        \let\SaveBegEnd=\BeginEnvironment%
\else
        % Start this environment
                \BeginEnvironment{#1}%
        % and execute code after \begin{environment}
                \csname AfterBeg#1\endcsname
        \let\SaveBegEnd=\NullCom
\fi
\SaveBegEnd{#1}
}
\def\end#1{%
% execute code before \end{environment}
\IfUnDef{BeforeEnd#1}\else\csname BeforeEnd#1\endcsname\fi%
% close this environment
\EndEnvironment{#1}
% and execute code after \begin{environment}
\IfUnDef{AfterEnd#1}\else\csname AfterEnd#1\endcsname\fi%
}
\long\def\BeforeBegin#1#2{\expandafter\gdef\csname BeforeBeg#1\endcsname {#2}}
\long\def\BeforeEnd  #1#2{\expandafter\gdef\csname BeforeEnd#1\endcsname {#2}}
\long\def\AfterBegin #1#2{\expandafter\gdef\csname AfterBeg#1\endcsname {#2}}
\long\def\AfterEnd   #1#2{\expandafter\gdef\csname AfterEnd#1\endcsname{#2}}

%%%% end of atbeginend.sty

%  ****************************************
%  *            PAGE LAYOUT               *
%  ****************************************

% Page size, spacing parameters, etc. 

\textwidth=7.1in
\columnsep=0.3125in
\parindent=0.125in
\voffset=-20mm
\hoffset=-7.5mm

\topmargin=0in
\headheight=.15in
\headsep=0.5in
\oddsidemargin=0in
\evensidemargin=0in
\parskip=0cm

\tolerance=600          % 3x "normal" value; cuts down on overfull complaints

%% AV,MM, to have 64 lines per column, with textheight 25cm:
\textheight=64\baselineskip
\textheight=\baselinestretch\textheight
\ifdim\textheight>25.2cm\textheight=25.0cm\fi

%% so that the margin at the bottom of the page is about equal to margin at top. -- AH
\advance\textheight -54pt

\topskip\baselineskip
\maxdepth\baselineskip



\def\eqsecnum{
    \@newctr{equation}[section]
    \def\theequation{\hbox{\normalsize\arabic{section}-\arabic{equation}}}}

%% running heads:
\def\lefthead#1{\gdef\@versohead{#1}} \lefthead{\relax}
\def\righthead#1{\ifanonymous\gdef\@rectohead{Anonymous author(s)}\else\gdef\@rectohead{#1}\fi} \righthead{\relax}
\let\shorttitle\lefthead
\let\shortauthors\righthead

\def\ps@apjheads{\let\@mkboth\markboth
     \def\@evenfoot{}
    \def\@evenhead{\lower9mm\hbox to\textwidth{%
                     \rm\thepage\hfill\textsc{\@rectohead}\hfill}}}
    \def\@oddfoot{}
    \def\@oddhead{\lower9mm\hbox to\textwidth{
                     \hfil\rm\textsc{\@versohead}\hfil \rm\thepage}}

\pagestyle{apjheads}

\@twosidetrue

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    TITLE PAGE                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% October 2017, adding preprint number capability
\newcount\rptnum
\newcount\rptloopnum
\def\reportnum#1{\global\advance\rptnum by 1
\expandafter\gdef\csname report\the\rptnum\endcsname{\hbox to
\textwidth{\hskip.75\textwidth\relax#1\relax\hfill}}}

\long\def\frontmatter@title@above{
\ifrnaas
\vspace*{-\headsep}\vspace*{\headheight}
\footnotesize
\noindent{\sc Draft version \today}\\[2pt]
{\footnotesize Typeset using \LaTeX\ {\bf RNAAS} style in AASTeX631}
\par\vspace*{-\baselineskip}\vspace*{0.625in}
\else
\vbox to 0pt{\vskip-48pt\normalsize\rptloopnum=0\loop
\global\advance\rptloopnum by 1
\expandafter\ifx\csname report\the\rptloopnum\endcsname\relax
\else
\csname report\the\rptloopnum\endcsname
\vskip1pt
\repeat
\vss
}
\ifmodern
\vglue-18pt
{\footnotesize
\raggedright
{\sc Draft version \today}\\[2pt]
{\footnotesize
Typeset using \LaTeX\ {\bf modern} style in AASTeX631}
\vskip48pt
}
\else
%%
\ifpreprint
\vspace*{-\headsep}\vspace*{\headheight}
\footnotesize
\noindent{\sc Draft version \today}\\[2pt]
{\footnotesize Typeset using \LaTeX\ {\bf preprint} style in AASTeX631}
\par\vspace*{-\baselineskip}\vspace*{0.625in}
\else
%%
\ifpreprinttwo
\vspace*{-\headsep}\vspace*{\headheight}
\footnotesize
{\footnotesize\textsc{\@journalinfo}}\par
{\footnotesize Typeset using \LaTeX\ {\bf preprint2} style in
AASTeX631}
\par\vspace*{-\baselineskip}\vspace*{0.625in}
\else
%%
\iftwocolstyle
\vspace*{-\headsep}\vspace*{\headheight}
\footnotesize
{\footnotesize\textsc{\@journalinfo}}\par
{\footnotesize Typeset using \LaTeX\ {\bf twocolumn} style in
AASTeX631}
\par\vspace*{-\baselineskip}\vspace*{0.625in}
\else
\ifmanu
\vspace*{-\headsep}\vspace*{\headheight}
\footnotesize
\noindent\textsc{\@journalinfo}\\[-8pt]
{\footnotesize Typeset using \LaTeX\ {\bf manuscript} style in
AASTeX631}
\par\vspace*{-\baselineskip}\vspace*{0.625in}
\else
%% Default
\vspace*{-\headsep}\vspace*{\headheight}
\footnotesize
{\footnotesize\textsc{\@journalinfo}}\par
{\footnotesize Typeset using \LaTeX\ default style in AASTeX631}%% default style
\par\vspace*{-\baselineskip}\vspace*{0.625in}
\fi\fi\fi\fi\fi
%%
\fi %% ends test to see if rnaas option was used; if rnaas is true
    %% none of this command will be activated.
}%

\def\frontmatter@title@produce{%
 \begingroup
\ifnumlines\nolinenumbers\fi
 \frontmatter@title@above
  \frontmatter@title@format
  \@title
%% October 2017, was +12pt
  \@ifx{\@title@aux\@title@aux@cleared}{}{%
\ifanonymous\else%% added may 20, 2020
  \expandafter\frontmatter@footnote\expandafter{\@title@aux}%
\fi
  }%
  \par
%% \class@warn{Unused short title ignored}%
\ifmodern
\vskip-42pt
\else
\ifmanu
\vskip-60pt
\else
  \frontmatter@title@below
\fi
\fi
 \endgroup
}%

\newcount\fncount
\newcount\endfncount
\long\def\tempfootnote#1{\global\advance\fncount by 1\relax%
\expandafter\gdef\csname
footnote\the\fncount\endcsname{\if@two@col\hsize=.5\textwidth
\advance\hsize by -18pt
\fi\relax#1}}

\def\tempfootnoteintitle#1{}

\long\def\ltx@foottext#1#2{%
 \begingroup
\expandafter\ltx@make@current@footnote\expandafter{\@mpfn}{#1}%
\@footnotetext{\vtop{\iftwocolstyle\hsize=.5\textwidth
\advance\hsize-18pt
\fi #2\vskip2pt}}% fixed for two col style, march 15, 2019
 \endgroup
}%


\def\kickoutfootnotes{\c@footnote=0\relax%
\loop\advance\endfncount by 1\relax%
\expandafter\ifx\csname footnote\the\endfncount\endcsname\relax%
\else%
\def\thempfn{\fnsymbol{footnote}}%
\advance\c@footnote by 1\relax%
\unskip\footnote{\csname footnote\the\endfncount\endcsname}%
\repeat}

%% 
\renewcommand\title[2][]{%
\def\@title{\setbox0=\vbox{\let\footnote\tempfootnote #2}%
{\let\footnote\tempfootnoteintitle%\uppercase
{#2}}%% right place for kickoutfootnotes below
\kickoutfootnotes
\ifdoublespace\vskip 3\baselineskip\fi
}%
 \def\@shorttitle{#1}%
 \let\@AF@join\@title@join
}%

\def\frontmatter@title@format{
%% No exception for rnaas, May, 2019
%\ifrnaas
%\large\scshape\centering
%\else
\normalsize
%% added Mar 2019
\bf\centering
%\fi
}

\def\frontmatter@title@below{\vskip8pt}%


\def\frontmatter@authorformat{\small\advance\baselineskip-3pt
\parskip=0pt\ifrnaas\else\scshape\fi
\ifmodern
\vskip48pt
\centering
\leftskip=.05in plus 1fil
\rightskip=.05in plus 1 fil
\baselineskip=13pt
\else
\centering
\fi
}%


\def\frontmatter@affiliationfont{\normalfont\footnotesize
%% changed october 2017
\it
\iflongauthor\else
\ifrnaas\else
\rightskip-12pt plus 1fil
\leftskip6pt \parindent-4pt
\fi\fi
}%

 \def\frontmatter@finalspace{\ifrnaas\vspace*{-\baselineskip}\vspace*{0.0in}
\else
\vskip-24pt\fi}

\def\@separator{\\}%

\newcounter{front@matter@foot@note}\setcounter{front@matter@foot@note}{0}

\iflongauthor%
\renewcommand\affiliation[1]{%
\parskip=2pt
\def\@centercr{\vrule depth 3pt width0pt\vskip1sp}
 \move@AU\move@AF%
\iflongauthor\let\@listand\relax\fi
 \begingroup%
  \@affiliation{%\hspace*{2mm}
\ifnum\countauthors<\AuthorCollaborationLimit
 #1\ifmodern\iflongauthor\baselineskip=12pt\else\vskip2pt\fi\else\baselineskip=11pt
\fi\fi
\ifnum\countauthors=\AuthorCollaborationLimit
 #1\ifmodern\iflongauthor\baselineskip=12pt\else\vskip2pt\fi\else\baselineskip=11pt
\fi\fi
\iflongauthor \vskip-10pt\ifnum\countauthors>\AuthorCollaborationLimit\iftwelvepoint\vskip-3pt\else\iftwocolstyle\else\vskip-6pt\fi\fi\fi%
\fi
}%
\let\@centercr\savecentercr}%
\else
%% older version
\renewcommand\affiliation[1]{%
 \move@AU\move@AF%
 \begingroup%
  \@affiliation{%\hspace*{2mm}
%% october 2017 changes:
\centering
 #1\ifmodern
\baselineskip=9.5pt
 \else
        \iftwelvepoint
	\ifpreprint\baselineskip=12pt\else\ifpreprinttwo\else\baselineskip19pt\fi\fi\else \baselineskip=9.5pt\fi
\fi}%
}%
\fi %% end switch for longauthor

%   ABSTRACT
\def\frontmatter@abstractfont{\normalsize\parindent=9pt
}%
%% october 2017
\def\frontmatter@abstractwidth{478pt}
\def\frontmatter@preabstractspace{12pt}
\def\frontmatter@postabstractspace{12pt}

\def\abstractname{ABSTRACT}
\long\def\frontmatter@abstractheading{%
 \begingroup
  \centering
\ifmodern\else\hskip34pt \fi \abstractname
  \vskip 1mm
  \par
 \endgroup
}%

\newif\ifabstract%
\renewenvironment{frontmatter@abstract}{%
  \aftermaketitle@chk{\begin{abstract}}%
\global\abstracttrue
  \global\setbox\absbox\vbox\bgroup
   \color@begingroup
%% width of abstract changed oct 2017
\ifmodern
 \columnwidth\textwidth
 \hsize\columnwidth
\else
\hsize = 478pt
\fi
   \@parboxrestore
   \def\@mpfn{mpfootnote}\def\thempfn{\thempfootnote)}\c@mpfootnote\z@ % should be footnote, not mpfootnote?
%   \let\@footnotetext\frontmatter@footnotetext
   \minipagefootnote@init
   \let\set@listindent\set@listindent@
   \let\@listdepth\@mplistdepth \@mplistdepth\z@
   \let@environment{description}{frontmatter@description}%
   \@minipagerestore
   \@setminipage
    \frontmatter@abstractheading
    \frontmatter@abstractfont
%    \let\footnote\mini@note
\ifmodern\else\everypar={\leftskip=34pt}\fi
%    \expandafter\everypar\expandafter{\the\everypar\addcontents@abstract\everypar{}}%
\ifnumlines\let\go\linenumbers\else\let\go\relax\fi\go
}{%
\ifnumlines\let\go\endlinenumbers\else\let\go\relax\fi\go
    \par
    \unskip
    \minipagefootnote@here
    \@minipagefalse   %% added 24 May 89
    \color@endgroup
\egroup% end setbox\absbox
}%

\let\abstract\frontmatter@abstract

\def\frontmatter@abstract@produce{%
  \par
  \preprintsty@sw{%
   \do@output@MVL{%
    \vskip\frontmatter@preabstractspace
    \vskip200\p@\@plus1fil
    \penalty-200\relax
    \vskip-200\p@\@plus-1fil
   }%
  }{%
   \addvspace{\frontmatter@preabstractspace}%
  }%
   \begingroup
    \dimen@\baselineskip
    \setbox\z@\vtop{\unvcopy\absbox}%
    \advance\dimen@-\ht\z@\advance\dimen@-\prevdepth
    \@ifdim{\dimen@>\z@}{\vskip\dimen@}{}%
   \endgroup
   \begingroup
   \prep@absbox
%% centered, oct 2017
\unvbox\absbox
    \post@absbox
   \endgroup
  \@ifx{\@empty\mini@notes}{}{\mini@notes\par}%
  \addvspace\frontmatter@postabstractspace
\global\abstractfalse
\vskip12pt
}%

\newif\ifbib

%  ****************************************
%  *             KEYWORDS                 *
%  ****************************************


  \def\@keys@name{\textit{Keywords:}\/~\mbox{}}%

\newlength{\keys@width}
\def\frontmatter@keys@format{\ifmodern\vskip0pt\else\vspace*{0.5mm}\fi%
  \settowidth{\keys@width}{\normalsize\@keys@name}%
\ifmodern\else
\rightskip=0.5in
\leftskip=34pt
\fi
\parindent=0pt%
    \hangindent=\keys@width\hangafter=1\normalsize}%
\def\@keywords@produce#1{%
 \showKEYS@sw{%
  \begingroup%
   \frontmatter@keys@format%
   \@keys@name#1
\vrule depth 12pt width 0pt
\ifnobreakafterkeywords
\vrule depth 24pt width0pt\fi
\par
  \endgroup
 }{%
  \@if@empty{#1}{}{%
   \class@warn{If you want your keywords to appear in your output, use document class option showkeys}%
  }%
 }%
}%

\let\subjectheadings=\keywords


%  ****************************************
%  *             FOOTNOTES                *
%  ****************************************

%% legacy notes and footnote code

% Footnotes on the last page: user issues \lastpagefootnotes. It catches all
% footnotes and issues them before \begin{refernces} or
% \begin{\thebibliography} or \begin{appendix}, whichever comes first,
% at the end of right column.  
\newcounter{lastfootnote}
\let\orig@footnote=\footnote
%
\def\spit@out@footnotes@{\addtocounter{footnote}{-\c@lastfootnote}\vspace*{-\baselineskip}\vspace*{\skip\footins}\bgroup\footnotesize\lastfootnote\par\egroup\let\footnote=\orig@footnote}
%
\let\spit@out@footnotes\relax
%
\long\def\lastpagefootnotes{%
\raggedbottom% Have to use flushbottom with revtex4-1 but it eats away
             % last page footnotes
\setcounter{lastfootnote}{0}
\long\gdef\lastfootnote{\mbox{}}
\let\spit@out@footnotes\spit@out@footnotes@
\long\def\footnote##1{\refstepcounter{lastfootnote}\footnotemark\g@addto@macro\lastfootnote{\\[\footnotesep]\refstepcounter{footnote}\mbox{}\hspace*{3mm}\textsuperscript{\thefootnote}~##1}}
% We need to mess with footnotes in appendix only if it is in a
% different mode (twocol vs onecol) than the main text
\def\mess@with@appendix@footnotes{
\BeforeBegin{appendix}{\spit@out@footnotes\let\spit@out@footnotes\relax}
\BeforeEnd{appendix}{\spit@out@footnotes\let\spit@out@footnotes\relax}
\AfterEnd{appendix}{\let\footnote=\orig@footnote}
}

\if@two@col
  \if@two@col@app\else
    \mess@with@appendix@footnotes
  \fi
\fi

\BeforeBegin{references}{\spit@out@footnotes\let\spit@out@footnotes\relax}
\BeforeBegin{thebibliography}{\spit@out@footnotes\let\spit@out@footnotes\relax}
% in case \lastpagefootnotes were given inside \begin{appendix}\end{appendix},
\AfterEnd{references}{\let\footnote=\orig@footnote}
\AfterEnd{thebibliography}{\let\footnote=\orig@footnote}
}
\let\lastpagefootnote=\lastpagefootnotes

%% probably not needed:
\def\notetoeditor#1{}%   % We do not need notes to editor in the preprint
\def\placetable#1{}%   % We do not need notes to editor in the preprint
\def\placefigure#1{}%   % We do not need notes to editor in the preprint

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ifdoublespace %used for manuscript
\footnotesep 12pt
\skip\footins 36pt
\else
\footnotesep 12pt
\skip\footins 24pt
\fi

\let\footnoterule\relax

\def\@makefntext#1{\hsize=\columnwidth\mbox{}\hspace*{3mm}\@makefnmark~#1}


%  ****************************************
%  *             SECTIONS                 *
%  ****************************************

\setcounter{secnumdepth}{3}

\newcount\tempsectnum

\newif\if@firstsection \@firstsectiontrue


\def\section{%
\ifanonymous\nocollaboration{0}\fi
\if@firstsection 
%% In case no collaboration is listed:
\ifcollaborationon\else\let\doauthor\olddoauthor
\ifanonymous \else
\let\allauthors=\oldallauthors\fi
\fi
%\edef\currauthorlimit{\the\AuthorCollaborationLimit}
%\collaboration{\currauthorlimit}{}
     \maketitle
\global\@firstsectionfalse
     \setcounter{footnote}{\thefront@matter@foot@note}%
     \let\footnotetext=\old@foot@note@text
     \let\footnotemark=\old@foot@note@mark
\ifnobreakafterkeywords
\ifmodern\vskip24pt
\else
\vskip12pt\fi%%
\goodbreak
\else
\ifrnaas\else
 \clearpage\fi
\fi
   \if@two@col
   \twocolumngrid
   \fi
\fi
\tempsectnum=\the\c@section
\advance\tempsectnum by 1
\xdef\cref@currentlabel{section \the\tempsectnum}
    \@startsection{section}{1}{\z@}{9pt plus 1pt minus
    1pt}{4pt}{\apjsecfont\center}} 

\def\subsection{\@startsection{subsection}{2}{\z@}{9pt plus 1pt minus 1pt}{4pt}%
  {\normalsize\itshape \center}}

\def\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {2ex plus 1ex minus .2ex}{1ex plus .2ex}{\small\itshape \center}}

\def\paragraph{\@startsection{paragraph}{4}{\z@}%
  {1.5ex plus 1ex minus .2ex}{0pt}{\small\itshape}}

\def\subparagraph{\@startsection{subparagraph}{5}{\z@}%
  {1ex plus 1ex minus .2ex}{-0.5\parindent}{\small\itshape}}


\def\thesection{\arabic{section}}
\def\thesubsection{\thesection.\arabic{subsection}}
\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\def\theparagraph{\thesubsubsection.\arabic{paragraph}}
\def\p@section        {}
\def\p@subsection     {}
\def\p@subsubsection  {}
\def\p@paragraph      {}
\def\p@subparagraph   {}


\def\sec@upcase#1{\uppercase{#1}}
\def\subsec@upcase#1{\relax{#1}}

%
% How the section number will appear in the section title - AV
\def\ApjSectionMarkInTitle#1{\ifrnaas\else #1.\ \fi}
\def\ApjSectionpenalty{0}


\def\@sect#1#2#3#4#5#6[#7]#8%
{\ifnum#2=1\setbox0=\hbox{\def\label##1{\gdef\templabel{##1}}#7}\fi
\@tempskipa #5\relax 
 \ifdim \@tempskipa >\z@ \begingroup
     #6\relax 
  \ifnum #2>\c@secnumdepth \def \@svsec {}\else 
    \refstepcounter{#1} \edef \@svsec {\ApjSectionMarkInTitle 
    {\csname the#1\endcsname}}\fi
  \@hangfrom {\hskip #3\relax 
    \ifnum #2=1{\secnum@size {\rm\@svsec~}}%
    \else {\subsecnum@size {\rm\@svsec~}}\fi }%
  {\interlinepenalty \@M 
   \ifnum #2=1\sec@upcase{#8}%
   \else \subsec@upcase{#8}\fi\par}\endgroup 
  \csname #1mark\endcsname {#7}\addcontentsline{toc}{#1}%
  {\ifnum #2>\c@secnumdepth \else \protect \numberline 
     {\csname the#1\endcsname }\fi #7}%
  \else 
  \ifnum #2>\c@secnumdepth \def \@svsec {}\else 
    \refstepcounter{#1} \edef \@svsec {\ApjSectionMarkInTitle 
    {\csname the#1\endcsname}}\fi
    \def \@svsechd {#6\hskip #3%
    \ifnum #2=1{\secnum@size{\rm\@svsec~}}\else{\subsecnum@size{\rm\@svsec~}}\fi%
    \ifnum #2=1\sec@upcase{#8}\else\subsec@upcase{#8}\fi%
    \ifnum #2=4\hskip 0.4ex{\rm ---}\fi%
     \csname #1mark\endcsname {#7}\addcontentsline{toc}{#1}%
      {\ifnum #2>\c@secnumdepth \else \protect \numberline {\csname
      the#1\endcsname }\fi #7}}\fi \@xsect {#5} \penalty \ApjSectionpenalty
%-refstepcounter is now within a group. So \@currentlabel, which is normally
%-set by \refstepcounter is hidden within a group. Need to set it manually. 
\protected@edef\@currentlabel{\csname p@#1\endcsname\csname
the#1\endcsname} 
\expandafter\ifx\csname templabel\endcsname\relax
\let\go\relax
\else
\def\go{\label{\templabel}}\fi\go
\let\templabel\relax
}

\def\@ssect#1#2#3#4#5{\@tempskipa #3\relax
   \ifdim \@tempskipa>\z@
     \begingroup #4\@hangfrom{\hskip #1}{\interlinepenalty \@M
       \sec@upcase{#5}\par}\endgroup
   \else \def\@svsechd{#4\hskip #1\relax \sec@upcase{#5}}\fi
    \@xsect{#3}
% MM:
\penalty \ApjSectionpenalty}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% changed to begin...end form, May 20,2020, AH
%% Acknowledgments, use \begin{acknowledgments}...\end{acknowledgments}
%% or \begin{acknowledgements}...\end{acknowledgements}

{\catcode`#=12
\gdef\hashmark{#}}

\def\xacknowledgments{acknowledgments}
\def\xacknowledgements{acknowledgements}
\newbox\ackbox

\def\acknowledgments{%
\ifx\@currenvir\xacknowledgments\let\go\relax\else
\let\go\stop
\typeout{x^^J^^J^^J^^J^^J^^J^^J^^J
 ===================^^J
In AASTeX v6.3.1 the \string\acknowledgments\space command has been deprecated.^^J^^J
Instead, please use the begin/end form:^^J^^J
"\string\begin\string{acknowledgments\string}...\string\end\string{acknowledgments\string}"^^J^^J 
when using acknowledgments.^^J^^J See here:
\url{https://journals.aas.org/aastexguide/\hashmark acknowledgments}
^^J^^J
 for more
details.^^J
 ===================^^J
}%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Printed error
\vskip12pt
\hrule height 1pt
\vskip12pt
\noindent ERROR:
In AASTeX v6.3.1 the {\tt\string\acknowledgments} command has been
deprecated.\\
Instead, please use the begin/end form:\\
{\tt
\string\begin\string{acknowledgments\string}...\string\end\string{acknowledgments\string}}\\
when using acknowledgments. For more details, see:\\ 
\url{https://journals.aas.org/aastexguide/\hashmark acknowledgments}
%
\fi\go
%%
\global\setbox\ackbox=\vbox\bgroup
\expandafter\ifx\csname internallinenumbers\endcsname\relax\else
\begin{internallinenumbers}
\fi
\vskip 5.8mm plus 1mm minus 1mm
\vskip1sp
%\section*{Acknowledgments}
\noindent\ignorespaces}

\def\endacknowledgments{
\expandafter\ifx\csname internallinenumbers\endcsname\relax\else
\end{internallinenumbers}
\fi
\egroup%% completes ackbox
\ifanonymous
\vskip 5.8mm plus 1mm minus 1mm
\vskip1sp
%\section*{Acknowledgments}
\centerline{(Acknowledgments anonymized for review)}
\else
\vbox{\unvbox\ackbox}
\fi\vskip6pt}

%% alternate spelling
\def\acknowledgements{%
\ifx\@currenvir\xacknowledgements\let\go\relax\else
\let\go\stop
\typeout{x^^J^^J^^J^^J^^J^^J^^J^^J
 ===================^^J
In AASTeX v6.3.1 the \string\acknowledgements\space command has been deprecated.^^J^^J
Please use the begin/end form:^^J^^J
"\string\begin\string{acknowledgements\string}...\string\end\string{acknowledgements\string}"^^J^^J 
when using acknowledgements.^^J^^J See here:
\href{https://journals.aas.org/aastexguide/\hashmark acknowledgments}
{https://journals.aas.org/aastexguide/\hashmark acknowledgments}
^^J^^J
 for more
details.^^J
 ===================^^J
}%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Printed error
\vskip12pt
\hrule height 1pt
\vskip12pt
\noindent ERROR:
In AASTeX v6.3.1 the {\tt\string\acknowledgements} command has been
deprecated.\\
Instead, please use the begin/end form:\\
{\tt
\string\begin\string{acknowledgements\string}...\string\end\string{acknowledgements\string}}\\
For more details, see:\\ 
\url{https://journals.aas.org/aastexguide/\hashmark acknowledgments}
%
\fi\go
%%
\global\setbox\ackbox=\vbox\bgroup
\expandafter\ifx\csname internallinenumbers\endcsname\relax\else
\begin{internallinenumbers}
\fi
\vskip 5.8mm plus 1mm minus 1mm
\vskip1sp
%\section*{Acknowledgments}
\noindent\ignorespaces}

\def\endacknowledgements{
\expandafter\ifx\csname internallinenumbers\endcsname\relax\else
\end{internallinenumbers}
\fi
\egroup%% completes ackbox
\ifanonymous
\vskip 5.8mm plus 1mm minus 1mm
\vskip1sp
%\section*{Acknowledgements}
\centerline{(Acknowledgements anonymized for review)}
\else
\vbox{\unvbox\ackbox}
\fi\vskip6pt}






%  ****************************************
%  *          APPENDIX                    *
%  ****************************************

\newcounter{remember@figure@num}
\newcounter{remember@table@num}

% this was an environment earlier, which doesn't make sense since we don't
% do \begin{appendix}...\end{appendix}. Changed to \appendix which is how it is used.

%   numberedappendix: Needed to label appendix sections A, B, .
%   appendixfloats: Needed. ??

\newif\ifappendixon
\def\appendix{\global\appendixontrue
\if@two@col  
\onecolumngrid
\noindent\mbox{}\vrule height 24pt width0pt\hfill{\apjsecfont APPENDIX}\hfill\mbox{}\par
\vskip18pt
   \if@two@col@app\global\@two@coltrue\twocolumngrid \fi
   % above, we want onecolumngrid to be default. Only twocolumn is asked for in documentclass option
\else
\noindent\mbox{}\vrule height 24pt width0pt\hfill{\apjsecfont
APPENDIX}\hfill\mbox{}\par 
\vskip18pt
    \if@two@col@app\global\@two@coltrue\twocolumngrid 
         \fi\fi
% \vrule used for extra space; otherwise revtex4-1 sometimes eats
% away the last line before appendix
        \nopagebreak\medskip\@nobreaktrue\def\ApjSectionpenalty{\@M}
        \@firstsectionfalse
          \setcounter{section}{0}
          \setcounter{subsection}{0}
          \setcounter{equation}{0}
%%
\ifletteredappendix
          \def\thesection{\Alph{section}}
          \def\theequation{\hbox{\Alph{section}\arabic{equation}}}
%% these are NOT supposed to reset to zero
%\setcounter{table}{0}
%\setcounter{figure}{0}
\fi
%%
\if@number@appendix@floats
% we don't want to set equations to zero in appendix, because there
% might then be two equation 1's, etc., confusing for cross referencing.
%\setcounter{equation}{0}
          \def\thesection{\Alph{section}}
          \def\theequation{\hbox{\Alph{section}\arabic{equation}}}
          \def\section{\@startsection {section}{1}{\z@} 
            {10pt}{5pt}{\centering\scshape\apjsecfont}}
\else
% Do not use appendix numbers in the titles
          \def\ApjSectionMarkInTitle{\AppendixApjSectionMarkInTitle}
\fi
\ifappletter
\let\savesection\section
\def\section{\resetapptablenumbers\savesection}
\fi
}
%


%%

%  ****************************************
%  *          BIBLIOGRAPHY                *
%  ****************************************

\renewenvironment{references}[0]{
  \onecolumngrid
  \par
  \vspace{10pt plus 3pt}
  \noindent \makebox[\textwidth][c]{\small\scshape REFERENCES}
  \par
  \vspace*{4pt plus 3pt}
  \set@column@hsize{2}\twocolumngrid 
%% Jan29, 2021, added \ifonecolstyle, \ifpreprinttwo
\ifnumlines
\ifonecolstyle
\ifpreprinttwo\else
\advance\linenumbersep-12pt\fi\fi\fi
  \parindent=0cm \parskip=0cm
  \def\baselinestretch{1.0}
  \footnotesize \baselineskip=\baselinestretch\bibbaselineskip plus 1pt
  minus 1pt \interlinepenalty \@M
  \hyphenpenalty=10000
  \frenchspacing    % AV - to get right spacing after ``et al.'' 
  \def\refpar{\par\hangindent=1.2em\hangafter=1}}
{
  \onecolumngrid
}

% redefine thebibliography

% remove numbers from the reference list
\def\@biblabel#1{\relax}

\newskip\bibskip
\bibskip=0pt plus 1pt
\let\savebibitem\bibitem
\def\bibitem{\vskip\bibskip\savebibitem}
\newdimen\bibindent
\renewenvironment{thebibliography}[1]{\global\bibtrue
%%\ifrnaas\newpage\fi% Not wanted, March 2019
\onecolumngrid
\vspace{20pt}
\goodbreak
    \hbox to\textwidth{\hss\normalsize REFERENCES\hss}
\vspace{6pt}\parskip=0pt
\twocolumngrid 
%% Jan29, 2021, added \ifonecolstyle
\ifnumlines
\ifonecolstyle
\ifpreprinttwo\else
\advance\linenumbersep-12pt\fi\fi\fi
\par
 \raggedright
\small
\ifmodern\else
 \vspace{10pt plus 3pt}\fi
\par
\topsep=0pt
 \list{}%
   {
     \parindent=0pt \parskip=1pt plus 1pt \parsep=0pt % AV
     \bibindent=0pt                          %
\ifmodern\vskip-12pt
\baselineskip=13pt plus 1pt
\else
\ifdoublespace
\baselineskip=20pt
\else
\baselineskip=13pt plus 1pt \fi\fi \interlinepenalty \@M  % AV
     \frenchspacing    % AV - to get right spacing after ``et al.'' 
     \hyphenpenalty=10000
     \itemindent=-1.0em                      %
     \itemsep=0pt                            %
     \listparindent=0pt                      %
     \settowidth\labelwidth{0pt} %
     \labelsep=0pt                           %
     \leftmargin=1.0em
     \advance\leftmargin\labelsep
%%%      \usecounter{enumiv}%
      \let\p@enumiv\@empty
      \renewcommand\theenumiv{\relax}}%
    \sloppy\clubpenalty10000\widowpenalty10000%
    \sfcode`\.\@m\relax
%%%\item[] This May be necessary, but causes extra vertical space in first column
%%%          of the bibliography.
}
  {\def\@noitemerr
    {\@latex@warning{Empty `thebibliography' environment}}%
\endlist
    \onecolumngrid % to balance references
\global\bibfalse
\newpage
  }

% %% AV: 

\def\reference{\@ifnextchar\bgroup {\@reference}
        {\@latexerr{Missing key on reference command}
        {Each reference command should have a key corresponding to a markcite somewhere in the text}}}
\def\@reference#1{\refpar}

%% subtitle header and journal info, legacy, probably not needed
% \def\submitted#1{\gdef\@submitted{#1}}
% \let\slugcomment\submitted   % slugcomment == submitted
% \submitted{Draft version \today}
% \journalinfo{\@submitted}

 \def\journalinfo#1{\gdef\@journalinfo{#1}}
 \journalinfo{Draft version \today}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Equations                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EQNARRAY with reduced spacing around tab characters - AV
\def\eqnarray{%
  \stepcounter{equation}%
  \def\@currentlabel{\p@equation\theequation}%
  \global\@eqnswtrue
  \setlength{\arraycolsep}{0.25\arraycolsep}
  \m@th
  \global\@eqcnt\z@
  \tabskip\@centering
  \let\\\@eqncr
  $$\everycr{}\halign to\displaywidth\bgroup
  \hskip\@centering$\displaystyle\tabskip\z@skip{##}$\@eqnsel
  &\global\@eqcnt\@ne\hskip \tw@\arraycolsep \hfil${##}$\hfil
  &\global\@eqcnt\tw@ \hskip \tw@\arraycolsep
  $\displaystyle{##}$\hfil\tabskip\@centering
  &\global\@eqcnt\thr@@ \hb@xt@\z@\bgroup\hss##\egroup
  \tabskip\z@skip
  \cr
  }

%  ****************************************
%  *         TABLES AND FIGURES           *
%  ****************************************

\def\@arstrut@hline@clnc{0.5\p@}% % Klootch: magic number

\setcounter{topnumber}{7}

\setlength{\belowcaptionskip}{4pt}% 
\setlength{\abovecaptionskip}{5pt}

\setlength{\textfloatsep}{9pt}
\setlength{\dbltextfloatsep}{0pt}

%  \addtolength{\textfloatsep}{17mm plus 10mm minus 10mm}
 % \addtolength{\dbltextfloatsep}{7mm plus 10mm minus 10mm}


\renewcommand{\topfraction}{1.0}
\renewcommand{\bottomfraction}{1.0}
\renewcommand{\textfraction}{0.0}
\renewcommand{\dbltopfraction}{0.85}
\renewcommand{\dblfloatpagefraction}{0.85}
\setcounter{topnumber}{9}
\setcounter{bottomnumber}{9}
\setcounter{totalnumber}{20}
\setcounter{dbltopnumber}{9} 

% Copied from revtex4.cls; without it, captions are centered
\def\@xfloat@prep{%
  \let\footnote\footnote@latex
  \def\@mpfn{mpfootnote}%
  \def\thempfn{\thempfootnote}%
  \c@mpfootnote\z@
  \let\@footnotetext\@mpfootnotetext
  \let\H@@footnotetext\@mpfootnotetext
  \let\@makefntext\@mpmakefntext
}%


%%%%%%%%%%%% \cutinhead

%% Number of \pt@ncol set when \tablecolumns{} used at beginning of
%% deluxetable.

\def\@ptabularcr{\\}
\newcommand\cutinhead[1]{% 
 \noalign{\vskip 1.5ex}% 
 \hline 
 \@ptabularcr 
 \noalign{\vskip -1.5ex}% 
 \multicolumn{\pt@ncol}{c}{#1}% 
 \@ptabularcr 
 \noalign{\vskip .8ex}% 
 \hline 
 \@ptabularcr 
 \noalign{\vskip -2ex}% 
}% 

\newcommand\sidehead[1]{% 
 \noalign{\vskip 1.5ex}% 
%  \hline 
%  \@ptabularcr 
%  \noalign{\vskip -1.5ex}% 
 \multicolumn{\pt@ncol}{l}{#1}% 
 \@ptabularcr 
 \noalign{\vskip .8ex}% 
%  \hline 
%  \@ptabularcr 
%  \noalign{\vskip -2ex}% 
}% 


%% these don't seem to be relevant:
\newcommand\figcaption{\@testopt{\@xfigcaption}{}}% 
%% suggested by Joern Wilms
\def\@figcaption#1{{\def\@captype{figure}\caption{\footnotesize #1}}}
\def\@xfigcaption[#1]#2{{\def\@captype{figure}\caption{\footnotesize
#2}}}

\def\thefigure{\@arabic\c@figure}

\def\fnum@figure{{\footnotesize{\@eapj@cap@font\rm
\mbox{\@eapj@figname~\thefigure}% <== was period here
}}}

\def\thetable{\@arabic\c@table}
%%% Table captions without making a floating table
\def\tabcaption{\@ifnextchar[{\@xtabcaption}{\@tabcaption}}
\def\@tabcaption#1{{\def\@captype{table}\caption{#1}}}
\def\@xtabcaption[#1]#2{{\def\@captype{table}\caption{#2}}}

%% redefined below
\def\fnum@table{{\@eapj@cap@font \@eapj@tabname~\thetable}}

 \let\fnum@ptable=\fnum@table
 \def\fnum@ptablecont{{\centering{\scshape Table~\thetable}---{\itshape
 Continued}}}% 
 \long\def\@make@caption@text#1#2{% 
   {\small\centering#1{\scshape #2}\par\vskip1.4ex}
 }% 

\long\def\@makecaption@plano@cont#1#2{% 
  {\small \centering#1\par}\vskip1.4ex\relax
}% 


%% Commands from aastex.cls:

% This has been redefined below
%\newcommand\dataset{\@testopt\@dataset{[}}%
%\def\@dataset[#1]#2{#2}%

\newcommand\facility{\@testopt\@facility{[}}%
\def\@facility[#1]#2{{\vskip6pt{\large\it Facility:} #2}}%

\newcommand\facilities{\@testopt\@facilities{[}}%
\def\@facilities[#1]#2{{\vskip6pt{\large\it Facilities:} #2}}%


\long\def\software{\bgroup\@testopt \@software {[}}
\def\@software[#1]#2{\vskip 6pt{
\frenchspacing
\iftwelvepoint
\font\foo=cmr12
\else
\font\foo=cmr10\fi
%%
\fontdimen2\foo=3pt %% Changed from 1.5pt to 3pt, March12, 2019
%%
{\large \it Software: }
#2
%% returning to original fontdimen
\iftwelvepoint
\fontdimen2\foo=3.91663pt
\else
\fontdimen2\foo=3.33333pt
\fi
}\egroup}





\newcommand\object{\@testopt\@object{[}}%
\def\@object[#1]#2{#2}%
\newcommand\objectname{\@testopt\@objectname{[]}}% 
\def\@objectname[#1]#2{#2}% 
\newlength{\plot@width}
\def\eps@scaling{1.0}% 
\newcommand\epsscale[1]{\def\eps@scaling{#1}}% 
\newcommand\plotone[1]{% 
 \centering 
 \leavevmode 
 \setlength{\plot@width}{0.85\linewidth}
 \includegraphics[width={\eps@scaling\plot@width}]{#1}% 
}% 
\newcommand\plottwo[2]{% 
 \centering 
 \leavevmode 
 \setlength{\plot@width}{0.425\linewidth}
 \includegraphics[width={\eps@scaling\plot@width}]{#1}% 
 \hfil 
 \includegraphics[width={\eps@scaling\plot@width}]{#2}% 
}% 
\newcommand\plotfiddle[7]{% 
 \centering 
 \leavevmode 
 \vbox\@to#2{\rule{\z@}{#2}}% 
 \includegraphics[% 
  scale=#4, 
  angle=#3, 
  origin=c 
 ]{#1}% 
}% 
\newcommand\figurenum[1]{% 
 \def\thefigure{#1}% 
 \let\@currentlabel\thefigure 
 \addtocounter{figure}{\m@ne}% 
}% 

\newcommand\phn{\phantom{0}}% 
\newcommand\phd{\phantom{.}}% 
\newcommand\phs{\phantom{$-$}}% 
\newcommand\phm[1]{\phantom{#1}}% 
\let\la=\lesssim            % For Springer A&A compliance... 
\let\ga=\gtrsim 
\newcommand\sq{\mbox{\rlap{$\sqcap$}$\sqcup$}}% 
\newcommand\arcdeg{\mbox{$^\circ$}}% 
\newcommand\arcmin{\mbox{$^\prime$}}% 
\newcommand\arcsec{\mbox{$^{\prime\prime}$}}% 
\newcommand\fd{\mbox{$.\!\!^{\mathrm d}$}}% 
\newcommand\fh{\mbox{$.\!\!^{\mathrm h}$}}% 
\newcommand\fm{\mbox{$.\!\!^{\mathrm m}$}}% 
\newcommand\fs{\mbox{$.\!\!^{\mathrm s}$}}% 
\newcommand\fdg{\mbox{$.\!\!^\circ$}}% 
\newcommand\farcm@mss{\mbox{$.\mkern-4mu^\prime$}}% 
\let\farcm\farcm@mss 
\newcommand\farcs@mss{\mbox{$.\!\!^{\prime\prime}$}}% 
\let\farcs\farcs@mss 
\newcommand\fp{\mbox{$.\!\!^{\scriptscriptstyle\mathrm p}$}}% 
\newcommand\micron{\mbox{$\mu$m}}% 
\def\farcm@apj{% 
 \mbox{.\kern -0.7ex\raisebox{.9ex}{\scriptsize$\prime$}}% 
}% 
\def\farcs@apj{% 
 \mbox{% 
  \kern  0.13ex.% 
  \kern -0.95ex\raisebox{.9ex}{\scriptsize$\prime\prime$}% 
  \kern -0.1ex% 
 }% 
}% 
 

\newcommand\ion[2]{#1$\;${%
\ifx\@currsize\normalsize\small \else
\ifx\@currsize\small\footnotesize \else
\ifx\@currsize\footnotesize\scriptsize \else
\ifx\@currsize\scriptsize\tiny \else
\ifx\@currsize\large\normalsize \else
\ifx\@currsize\Large\large
\fi\fi\fi\fi\fi\fi
\rmfamily\@Roman{#2}}\relax}% 

\renewcommand\case[2]{\mbox{$\frac{#1}{#2}$}}% 
\renewcommand\slantfrac{\case}% 
\newcommand\onehalf{\slantfrac{1}{2}}% 
\newcommand\onethird{\slantfrac{1}{3}}% 
\newcommand\twothirds{\slantfrac{2}{3}}% 
\newcommand\onequarter{\slantfrac{1}{4}}% 
\newcommand\threequarters{\slantfrac{3}{4}}% 
\newcommand\ubvr{\mbox{$U\!BV\!R$}}%% UBVR system 
\newcommand\ub{\mbox{$U\!-\!B$}}%   % U-B 
\newcommand\bv{\mbox{$B\!-\!V$}}%   % B-V 
\renewcommand\vr{\mbox{$V\!-\!R$}}%   % V-R 
\newcommand\ur{\mbox{$U\!-\!R$}}%   % U-R 

%% need this change so that it works correctly in tables:
{\catcode`\$=\active
\gdef\nodata{ ~$\cdots$~ }}% 

\newcommand\diameter{\ooalign{\hfil/\hfil\crcr\mathhexbox20D}}% 
\newcommand\degr{\arcdeg}% 
\newcommand\Sun{\sun}% 
\newcommand\Sol{\sun}% 
\newcommand\sun{\odot}% 
\newcommand\Mercury{\astro{\char1}}% Mercury symbol, "1" 
\newcommand\Venus{\astro{\char2}}% Venus symbol, "2" 
\newcommand\Earth{\earth}% 
\newcommand\Terra{\earth}% 
\newcommand\earth{\oplus}% 
\newcommand\Mars{\astro{\char4}}% Mars symbol, "4" 
\newcommand\Jupiter{\astro{\char5}}% Jupiter symbol, "5" 
\newcommand\Saturn{\astro{\char6}}% Saturn symbol, "6" 
\newcommand\Uranus{\astro{\char7}}% Uranus symbol, "7" 
\newcommand\Neptune{\astro{\char8}}% Neptune symbol, "8" 
\newcommand\Pluto{\astro{\char9}}% Pluo symbol, "9" 
\newcommand\Moon{\astro{\char10}}% Moon symbol, "M" 
\newcommand\Luna{\Moon}% 
\newcommand\Aries{\astro{\char11}}% 
\newcommand\VEq{\Aries}% vernal equinox (Aries) 
\newcommand\Taurus{\astro{\char12}}% 
\newcommand\Gemini{\astro{\char13}}% 
\newcommand\Cancer{\astro{\char14}}% 
\newcommand\Leo{\astro{\char15}}% 
\newcommand\Virgo{\astro{\char16}}% 
\newcommand\Libra{\astro{\char17}}% 
\newcommand\AEq{\Libra}% autumnal equinox (Libra) 
\newcommand\Scorpius{\astro{\char18}}% 
\newcommand\Sagittarius{\astro{\char19}}% 
\newcommand\Capricornus{\astro{\char20}}% 
\newcommand\Aquarius{\astro{\char21}}% 
\newcommand\Pisces{\astro{\char22}}% 
 

\newcommand\sbond{\chem@bnd{\@sbnd}}%
\newcommand\dbond{\chem@bnd{\@dbnd}}%
\newcommand\tbond{\chem@bnd{\@tbnd}}%
\def\chem@bnd#1{%
 {%
  \kern.1em\relax
  \setbox\z@\hbox{M}%
  \dimen@ii.8em\relax
  \p@=.1em\relax
  \dimen@.5\ht\z@\dimen@i-\dimen@
  \advance\dimen@1.5\p@\advance\dimen@i-1.0\p@
  #1%
  \kern.1em\relax
  }%
 }%
\def\@sbnd{%
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 }%
\def\@dbnd{%
 \advance\dimen@-0.5\p@\advance\dimen@i0.5\p@
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \hskip-\dimen@ii
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 }%
\def\@tbnd{%
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \hskip-\dimen@ii
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \hskip-\dimen@ii
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 }%

%\newcommand\anchor[2]{#2}% 
%\renewcommand\url{\@dblarg\@url}% 
%\def\@url[#1]{\anchor{#1}}% 

\let\jnl@style=\rmfamily 
\def\ref@jnl#1{{\jnl@style#1}}% 
\newcommand\aj{\ref@jnl{AJ}}%        % Astronomical Journal 
\newcommand\psj{\ref@jnl{PSJ}}%       % Planetary Science Journal
\newcommand\araa{\ref@jnl{ARA\&A}}%  % Annual Review of Astron and Astrophys 
\renewcommand\apj{\ref@jnl{ApJ}}%    % Astrophysical Journal 
\newcommand\apjl{\ref@jnl{ApJL}}     % Astrophysical Journal, Letters 
\newcommand\apjs{\ref@jnl{ApJS}}%    % Astrophysical Journal, Supplement 
\renewcommand\ao{\ref@jnl{ApOpt}}%   % Applied Optics 
\newcommand\apss{\ref@jnl{Ap\&SS}}%  % Astrophysics and Space Science 
\newcommand\aap{\ref@jnl{A\&A}}%     % Astronomy and Astrophysics 
\newcommand\aapr{\ref@jnl{A\&A~Rv}}%  % Astronomy and Astrophysics Reviews 
\newcommand\aaps{\ref@jnl{A\&AS}}%    % Astronomy and Astrophysics, Supplement 
\newcommand\azh{\ref@jnl{AZh}}%       % Astronomicheskii Zhurnal 
\newcommand\baas{\ref@jnl{BAAS}}%     % Bulletin of the AAS 
\newcommand\icarus{\ref@jnl{Icarus}}% % Icarus
\newcommand\jaavso{\ref@jnl{JAAVSO}}  % The Journal of the American Association of Variable Star Observers
\newcommand\jrasc{\ref@jnl{JRASC}}%   % Journal of the RAS of Canada 
\newcommand\memras{\ref@jnl{MmRAS}}%  % Memoirs of the RAS 
\newcommand\mnras{\ref@jnl{MNRAS}}%   % Monthly Notices of the RAS 
\renewcommand\pra{\ref@jnl{PhRvA}}% % Physical Review A: General Physics 
\renewcommand\prb{\ref@jnl{PhRvB}}% % Physical Review B: Solid State 
\renewcommand\prc{\ref@jnl{PhRvC}}% % Physical Review C 
\renewcommand\prd{\ref@jnl{PhRvD}}% % Physical Review D 
\renewcommand\pre{\ref@jnl{PhRvE}}% % Physical Review E 
\renewcommand\prl{\ref@jnl{PhRvL}}% % Physical Review Letters 
\newcommand\pasp{\ref@jnl{PASP}}%     % Publications of the ASP 
\newcommand\pasj{\ref@jnl{PASJ}}%     % Publications of the ASJ 
\newcommand\qjras{\ref@jnl{QJRAS}}%   % Quarterly Journal of the RAS 
\newcommand\skytel{\ref@jnl{S\&T}}%   % Sky and Telescope 
\newcommand\solphys{\ref@jnl{SoPh}}% % Solar Physics 
\newcommand\sovast{\ref@jnl{Soviet~Ast.}}% % Soviet Astronomy 
\newcommand\ssr{\ref@jnl{SSRv}}% % Space Science Reviews 
\newcommand\zap{\ref@jnl{ZA}}%       % Zeitschrift fuer Astrophysik 
\renewcommand\nat{\ref@jnl{Nature}}%  % Nature 
\newcommand\iaucirc{\ref@jnl{IAUC}}% % IAU Cirulars 
\newcommand\aplett{\ref@jnl{Astrophys.~Lett.}}%  % Astrophysics Letters 
\newcommand\apspr{\ref@jnl{Astrophys.~Space~Phys.~Res.}}% % Astrophysics Space Physics Research 
\newcommand\bain{\ref@jnl{BAN}}% % Bulletin Astronomical Institute of the Netherlands 
\newcommand\fcp{\ref@jnl{FCPh}}%   % Fundamental Cosmic Physics 
\newcommand\gca{\ref@jnl{GeoCoA}}% % Geochimica Cosmochimica Acta 
\newcommand\grl{\ref@jnl{Geophys.~Res.~Lett.}}%  % Geophysics Research Letters 
\renewcommand\jcp{\ref@jnl{JChPh}}%     % Journal of Chemical Physics 
\newcommand\jgr{\ref@jnl{J.~Geophys.~Res.}}%     % Journal of Geophysics Research 
\newcommand\jqsrt{\ref@jnl{JQSRT}}%   % Journal of Quantitiative Spectroscopy and Radiative Trasfer 
\newcommand\memsai{\ref@jnl{MmSAI}}% % Mem. Societa Astronomica Italiana 
\newcommand\nphysa{\ref@jnl{NuPhA}}%     % Nuclear Physics A 
\newcommand\physrep{\ref@jnl{PhR}}%       % Physics Reports 
\newcommand\physscr{\ref@jnl{PhyS}}%        % Physica Scripta 
\newcommand\planss{\ref@jnl{Planet.~Space~Sci.}}%  % Planetary Space Science 
\newcommand\procspie{\ref@jnl{Proc.~SPIE}}%      % Proceedings of the SPIE 

\newcommand\actaa{\ref@jnl{AcA}}%  % Acta Astronomica
\newcommand\caa{\ref@jnl{ChA\&A}}%  % Chinese Astronomy and Astrophysics
\newcommand\cjaa{\ref@jnl{ChJA\&A}}%  % Chinese Journal of Astronomy and Astrophysics
\newcommand\jcap{\ref@jnl{JCAP}}%  % Journal of Cosmology and Astroparticle Physics
\newcommand\na{\ref@jnl{NewA}}%  % New Astronomy
\newcommand\nar{\ref@jnl{NewAR}}%  % New Astronomy Review
\newcommand\pasa{\ref@jnl{PASA}}%  % Publications of the Astron. Soc. of Australia
\newcommand\rmxaa{\ref@jnl{RMxAA}}%  % Revista Mexicana de Astronomia y Astrofisica

%% added feb 9, 2016
\newcommand\maps{\ref@jnl{M\&PS}}% Meteoritics and Planetary Science
\newcommand\aas{\ref@jnl{AAS Meeting Abstracts}}% American Astronomical Society Meeting Abstracts
\newcommand\dps{\ref@jnl{AAS/DPS Meeting Abstracts}}% American Astronomical Society/Division for Planetary Sciences Meeting Abstracts



\let\astap=\aap 
\let\apjlett=\apjl 
\let\apjsupp=\apjs 
\let\applopt=\ao 

\newcommand\ulap[1]{\vbox\@to\z@{{\vss#1}}}% 
\newcommand\dlap[1]{\vbox\@to\z@{{#1\vss}}}% 


\newcounter{table@save}

%% March 25, 2019
%% Old v5.2 way, From Greg, This allows a number like 33N to be used
%% for a table, and the cross-references will work correctly
\newcommand\tablenum[1]{%
 \def\thetable{#1}%
 \xdef\@currentlabel{\thetable}
\global\advance\c@table-1\relax
}%

\let\savetablenum\tablenum

\def\tabletypesize#1{\gdef\currtabletypesize{#1}
\def\@table@type@size{#1}}%

\tabletypesize{\small}
\let\tablefontsize=\tabletypesize % for compatibility with old documents

\gdef\@tablecaption{}
\def\tablecaption#1{\gdef\@tablecaption{#1}}

\def\LT@endpbox{%
  \@finalstrut\@arstrutbox
  \egroup
  \the\LT@p@ftn
  \global\LT@p@ftn{}%
  \hfil}

%% Used for longtable
\def\LT@makecaption#1#2#3{%
  \LT@mcol\LT@cols c{\hbox to\z@{\hss\parbox[t]\LTcapwidth
{%
\xdef \@currentlabel{\thetable}
     \sbox\@tempboxa{\small #2. 
%% disable trackchanges commands here, so they aren't entered 2 times:
\let\added\relax
\let\deleted\relax
\let\replaced\relax
#3}%
     \ifdim\wd\@tempboxa>\hsize
      \small#2. #3%
     \else
       \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
     \fi
    \endgraf\vskip\baselineskip}%
  \hss}}
}%% 

\let\LT@makecaption@rtx=\LT@makecaption % to fight redefinition in Revtex-4.1

\def\xfigure{figure}
%% from book.cls/ used??
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
%% \small added to keep currtabletypesize from determining size of caption
  \sbox\@tempboxa{\small
%% disable trackchanges commands here, so they aren't entered 2 times:
\let\added\relax
\let\deleted\relax
\let\replaced\relax
{\bf #1.} #2}%
  \ifdim \wd\@tempboxa >\hsize
\small
{\bf    #1.} #2\par 
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfill\box\@tempboxa\hfill}%
  \fi
  \vskip\belowcaptionskip
}

\newdimen\@abovenoteskip
\newcommand\tablerefs[1]{\ifdim\@abovenoteskip=0pt\global\@abovenoteskip=10pt\fi
{\small\@tableref{\parfillskip\z@ plus1fil\relax #1\endgraf}}}%

\def\@tableref#1{%
 \par
 \vspace*{3ex}%
 {%\parbox{\pt@width} %%%%
 {\hskip1em\rm References. --- #1}\par}%
}%

%% march 2019, added \it to tablenotemark
\global\def\tablenotemark#1{{\normalfont\textsuperscript{\normalsize\it #1}}}
\global\def\tablenotetext#1#2{\footnotetext[#1]{\currtabletypesize\relax#2}}

%% redefined by AH below, since it wasn't working with tabular table
\global\def\tablenotetext#1#2{\vskip-8pt\vskip1sp\flushleft{\currtabletypesize
\noindent\hskip1em $^{#1}$ #2}\vskip1sp}

%% this version of \tablehead doesn't seem to be used, so set to \xyztablehead{}
\def\xyztablehead#1{\@table@not@headedfalse%
  \kill
  \caption{\\%
   \@tablecaption\gdef\@currentlabel{\thetable}(0)}
    \\\hline\hline%
  #1\vrule height 12pt depth 10pt width 0pt\relax 
\hskip\tabcolsep\\[.7ex]
  \hline\\[-1.5ex]
  \endfirsthead
  \caption[]{--- \emph{Continued}}\\
  \hline
  \hline\\[-1.7ex]
  #1\hskip\tabcolsep\\[.7ex]
  \hline\\[-1.5ex]
  \endhead
  \hline
  \endfoot%
}

\newif\if@table@not@headed

\newlength{\table@note@skip}
\setlength{\table@note@skip}{0.5ex}
\newlength{\deluxe@table@width}
\newlength{\@d@t@a}
\newcounter{deluxe@table@num}
\newdimen\LTcapwidth

%% \ignorespaces necessary ++== depth 6pt was depth 3pt, == added height 12pt, nov 2017
\def\colhead#1{\multicolumn{1}{c}{\vrule depth 6pt height 12pt width
0pt\relax#1}\ignorespaces}
\def\twocolhead#1{\multicolumn{2}{c}{\hss\vrule depth 6pt height 12pt width
0pt\relax#1\hss}\ignorespaces}
\def\nocolhead#1{\multicolumn{1}{h}{}\ignorespaces}
\def\dcolhead#1{\multicolumn{1}{c}{$\vrule depth 6pt height12pt
width0pt\relax#1$}\ignorespaces}

%\newcounter{LT@tables}
\def\tablewidth#1{%
  \ifdim#1=\z@
  \else
  \gdef\@d@t@@flag{1}
  \if@filesw\immediate\write\@auxout{%
   \gdef\expandafter\noexpand
   \csname deluxe@table@width@\romannumeral\c@LT@tables\endcsname
   {#1}}
  \fi
  \fi
}


\def\save@natural@width{%
      \ifnum\@d@t@@flag=0
        \setlength{\@d@t@a}{0pt}%
        \let\@d@t@b=\LT@entry%
        \def\LT@entry##1##2{\addtolength{\@d@t@a}{##2}%
        }%
        \expandafter\csname LT@\romannumeral\c@deluxe@table@num\endcsname
        \setlength{\@d@t@a}{-\@d@t@a}
        \tablewidth{\the\@d@t@a}
        \def\LT@entry{\@d@t@b}
      \fi
}

\def\lt@expand@linewidth@one{\setlength\LTleft{0pt}\setlength\LTright{0pt}}
\def\lt@expand@linewidth@two{@{\extracolsep{0pt plus 1filll}}}

\def\find@table@width{%
%%% set table width using aux file and command \tablewidth
    \setcounter{deluxe@table@num}{\c@LT@tables}
    \refstepcounter{deluxe@table@num}
    \expandafter\ifx\csname deluxe@table@width@\romannumeral\c@deluxe@table@num\endcsname\relax
      \def\@d@t@{0.999\linewidth}
    \else
      \edef\@d@t@{\expandafter\csname deluxe@table@width@\romannumeral\c@deluxe@table@num\endcsname}
    \fi
    \ifdim\@d@t@<\z@% then natural width is used
      \setlength{\deluxe@table@width}{-\@d@t@}
      \setlength{\LTcapwidth}{-\@d@t@}
      \def\lt@expand@linewidth{\relax}
      \def\lt@expand@linewidth@{}
    \else% we will enclose table in the minipage of the given width and make
         % longtable to span the full minipage width
      \ifdim\@d@t@>\z@\else\def\@d@t@{0.999\linewidth}\fi
      \setlength{\deluxe@table@width}{\@d@t@}
      \setlength{\LTcapwidth}{\@d@t@}
      \def\lt@expand@linewidth{\lt@expand@linewidth@one}
      \def\lt@expand@linewidth@{\lt@expand@linewidth@two}
    \fi
}

\newlength{\abovedeluxetableskip}
\newlength{\belowdeluxetableskip}
\setlength{\abovedeluxetableskip}{0pt}
\setlength{\belowdeluxetableskip}{0pt}
\setlength{\tabcolsep}{5pt}

\setlength\doublerulesep{1.5pt}
\newdimen\lastrowheight
\def\set@last@row@height{\setlength{\lastrowheight}{\ht\strutbox}\addtolength{\lastrowheight}{\dp\strutbox}\setlength{\lastrowheight}{-\arraystretch\lastrowheight}}

\let\tableline=\colrule % Revtex said: Command \tableline is obsolete; Use \colrule instead.. 

%% 
\newtoks\DT@p@ftn 
\global\def\xtablenotetext@DT#1#2{
  \edef\@tempa{\the\DT@p@ftn\noexpand\tablenotemark{#1}~}
  \global\DT@p@ftn\expandafter{\@tempa{\@table@type@size#2}\par}}%

%%% AH %%%%%%%%%%%%%%%%%%%%%
\let\savedollar$
\catcode`\$=\active
\let$\savedollar

%%

\def\resetdecimals{\global\let\zdoit\relax\global\let\ddoit\relax}

\def\tableheadfrac#1{}
\newcount\pt@column 
\newcount\pt@ncol 

\newcommand\tablecolumns[1]{% 
 \pt@column=#1\relax 
 \pt@ncol=#1\relax 
 \global\let\pt@addcol\@empty 
}% 

%% from older version, probably can delete:
\def\@tablecom#1{% 
 \vspace*{\table@note@skip}
\par 
{\parbox{\linewidth}{\hskip1em\rmfamily {\@eapj@cap@font Note}. --- #1}\par}% 
}% 
\def\@tableref#1{% 
 \vspace*{\table@note@skip}
\par 
{\parbox{\linewidth}{\hskip1em\rmfamily {\@eapj@cap@font References}. --- #1}\par}% 
}% 
\def\spew@tblnotes{% 
 \@ifx@empty\tblref@list{}{% 
  \@tablenotes{\tblref@list}%
  \vspace*{\table@note@skip}%
  \global\let\tblref@list\@empty 
 }% 
 \@ifx@empty\tblnote@list{}{% 
  \@tablenotes{\tblnote@list}% 
  \vspace*{\table@note@skip}%
  \global\let\tblnote@list\@empty
 }
 \the\DT@p@ftn%
}% 

%% for notes on emulateapj please see http://hea-www.harvard.edu/~alexey/emulateapj
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% End of code taken from emulateapj.cls %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% New as of august 2015:

%% These commands requested AAS
\newcommand{\noprint}[1]{}
\newcommand{\figsetstart}{{\bf Fig. Set} }
\newcommand{\figsetend}{}
\newcommand{\figsetgrpstart}{}
\newcommand{\figsetgrpend}{}
\newcommand{\figsetnum}[1]{{\bf #1.}}
\newcommand{\figsettitle}[1]{ {\bf #1}}
\newcommand{\figsetgrpnum}[1]{\noprint{#1}}
\newcommand{\figsetgrptitle}[1]{\noprint{#1}}
\newcommand{\figsetplot}[1]{\noprint{#1}}
\newcommand{\figsetgrpnote}[1]{\noprint{#1}}

\usepackage{url}
%% if we take away the xx before UrlBreaks we will get a url that breaks
%% at any letter or number. It might be better to break only at / however...
\expandafter\def\expandafter\xxUrlBreaks\expandafter{\UrlBreaks%  save the current one
  \do\a\do\b\do\c\do\d\do\e\do\f\do\g\do\h\do\i\do\j%
  \do\k\do\l\do\m\do\n\do\o\do\p\do\q\do\r\do\s\do\t%
  \do\u\do\v\do\w\do\x\do\y\do\z\do\A\do\B\do\C\do\D%
  \do\E\do\F\do\G\do\H\do\I\do\J\do\K\do\L\do\M\do\N%
  \do\O\do\P\do\Q\do\R\do\S\do\T\do\U\do\V\do\W\do\X%
  \do\Y\do\Z\do\1\do\2\do\3\do\4\do\5\do\6\do\7\do\8\do\9}


%% for tables continuing over pages
\usepackage{longtable}

%% for editing changes
\usepackage{xcolor}
% hyperref link defaults to "blue" (0000ff) as this matches our publisher produced pdf style
\definecolor{xlinkcolor}{cmyk}{1,1,0,0}


\PassOptionsToPackage{hyphens}{url}
%% In response to request from AAS
 \usepackage[bookmarks=true,         % show bookmarks bar?/ Changed March 22, 2019 for
                                     % improved accessibility
    pdfnewwindow=true,      % links in new window
   colorlinks=true,    % false: boxed links; true: colored links
  linkcolor=xlinkcolor,     % color of internal links
 citecolor=xlinkcolor,     % color of links to bibliography
filecolor=xlinkcolor,  % color of file links
urlcolor=xlinkcolor,      % color of external links
final=true,
 ]{hyperref}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Additions to  AASTeX by Amy Hendrickson, TeXnology Inc, August 17, 2015

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Table Tools, written by Amy Hendrickson
%% for American Astronomical Society
%% August 17, 2015
%% array package is necessary:
\usepackage{array}

%%%%%%%%
%% 1) h in table preamble will makes text in that column ignored.

%%%%%%%%
%% 2) uppercase C L or R will make those columns in math mode.

%%%%%%%%
%% 3) \begin{splittabular}{<preamble B preamble>}...\end{splittabular}
%%    `B' in the preamble will show where the table should be broken. It may
%%    be used once in a preamble to break the table into two parts, or 
%%    two times in a preamble to break the table into three parts.

%%    The resulting table will position left half of table above right half of table.

%%    Care should be taken with \multicolumn so that it won't be broken between
%%    the two parts of the table. You may need to shorten the \multicolumn argument
%%    for number of columns to be appropriate to the number of columns in
%%    the new split table.

%%     In this example, for instance, we start with:
%%        \begin{tabular}{ccccccc}
%%        \multicolumn{7}{c}{\textsc{Table 1}}
%%     and change it to:
%%        \begin{splittabular}{ccccBccc}
%%        \multicolumn{4}{c}{\textsc{Table 1}}\\
%% 
%%     If you'd like to have a table number in the second half of the table,
%%     you can ask for a second multicolumn command that will position at the
%%     top of the second half of the table:
%%
%%        \begin{splittabular}{ccccBccc}
%%        \multicolumn{4}{c}{\textsc{Table 1}}&\multicolumn{3}{c}{\textsc{Table 1, Continued}}\\
%%
%%     You can do the same for the caption, and give a Continued caption for
%%     the second half of the table. Originally:
%%        \multicolumn{7}{c}{\textsc{Log of \textit{HST} Observations for
%%         NGC~6388}}
%%     Now, changed so that we get a caption on both halves of the table:
%%        \multicolumn{4}{c}{\textsc{Log of \textit{HST} Observations for
%%         NGC~6388}}&\multicolumn{3}{c}{\textsc{Log of \textit{HST} Observations for
%%         NGC~6388, Continued}}\\
%%     
%%     Similarly, you can rearrange table notes to appear at the bottom of
%%     the appropriate half of the table. For instance, starting with this,
%%     which would make table notes both appear at the bottom of the top half
%%     of the table:
%%        \multicolumn{4}{l}{\small{$\dagger$ Average distance of data set from
%%            cluster center.}}\\
%%        \multicolumn{4}{l}{$^{\rm a}$SNAP program.}\\
%%     
%%     We may substitute the following:
%%        \multicolumn{4}{l}{$^{\rm a}$SNAP program.}&\multicolumn{3}{l}{\small{$\dagger$ Average distance of data set from
%%            cluster center.}}\\
%%     
%%     Which will produce the first endnote underneath the top half of the
%%     table, and the second underneath the bottom half.
%%     
%%     Another option would be to have both endnotes appear below the bottom
%%     half of the table. Easily done:
%%        &&&&\multicolumn{3}{l}{$^{\rm a}$SNAP program.}\\
%%        &&&&\multicolumn{3}{l}{\small{$\dagger$ Average distance of data set from
%%            cluster center.}}\\
%%     %%     %%     %%     %%     %%     %%     

%%%%%%%%
%% 4) \colnumbers will make line with column numbers automatic. It will work with splittabular
%%     and splitdeluxetable as well as tabular.
%%     To use: type in \colnumbers within the table whereever you'd like it to appear, typically
%%     underneath the column headers, before the lines of data.

%%%%%%%%
%% 5) Easy Decimal numbering
%%    How to make decimal numbers in tables line up on the period:

%%     Use D (for decimal column) in table preamble for every decimal number. The decimal
%%     numbers will use two columns, one for the left part of the decimal
%%     number and one for the right part.

%%     D may be used more than once in a table preamble.

%%     If you want to type in a column header  over the decimal
%%     numbers, please use \multicolumn2c{} to span both columns.

%%     After the column headers, to start decimal numbering, 
%%     type in \decimals in the body of the table.

%%     When entering decimal numbers
%%     remember to leave a space after the decimal number, before the following &.
%%     For instance: &22.3 &35.96 \\

%%     If you'd like an empty entry, please supply a period and a space: & . &. You
%%     will not see the period in the resulting table.

%%     example:
%    \begin{tabular}{rDD}
%     \hline
%     &&&\multicolumn2c{\bf More}\\
%     &\multicolumn2c{\bf Decimals}&\multicolumn2c{\bf Decimals}\\
%     \hline
%     \hline
%     \decimals
%     one& . &34.2 \\
%     two &567.0 &21345 \\
%     three&.0 &62.5 \\
%     four&245 &5034.349923 \\
%     five&21 & \\
%     six& &21.6 
%    \end{tabular}
%%
%    Decimal numbering works within \begin{splittabular}...\end{splittabular}
%    \begin{splitdeluxetable} and \end{splitdeluxetable} for tables broken into two
%    or three parts. \colnumbers will number the columns counting both sides 
%    of the decimal number as one column; \tablehead{\colhead{}...} will make
%    the column headers position above the two columns used for one decimal
%    number as well.
%
%    Decimal numbers will be in math mode so that plus and minus signs are printed
%    correctly, expressions like `$\pm$ 1.2' or `\pm 1.2', will both work
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\let\savedollar$
\catcode`\$=\active

\let\savetabular\tabular
\def\tabular{\catcode`\&=\active\relax\catcode`\$=\active\relax\hskip\movetableright
\savetabular}
\long\gdef\eatone{\setbox0=\hbox\bgroup\savedollar\let$\relax}
\gdef\endeatone{\savedollar\egroup\hskip-2\tabcolsep}

%% Hide, important because it allows us to split tables horizontally
\newcolumntype{h}{>\eatone c<\endeatone}

\newcolumntype{C}{>{\bgroup\savedollar\let$\relax}c<{\savedollar\egroup}}
\newcolumntype{L}{>{\bgroup\savedollar\let$\relax}l<{\savedollar\egroup}}
\newcolumntype{R}{>{\bgroup\savedollar\let$\relax}r<{\savedollar\egroup}}

\newcolumntype{B}{>\eatone c<\endeatone}  %% used for \splittabular to indicate break in
                                          %% two or three parts of table

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Split tables into 2 or 3 parts; stack the parts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Process table preamble

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Count number of columns in preamble

\newcount\colcount
\newcount\firstcolcount
\newcount\secondcolcount
\newcount\thirdcolcount
\newcount\columncount

%% \makeatother is necessary to keep @ from being treated as a letter
%% when counting the number of columns in table.
\makeatother
\def\xD{D}
\def\xaster{*}
\newcount\howmanyDs
\newcount\firsthowmanyDs
\newcount\secondhowmanyDs

\newif\ifD

\def\countcols#1{\ifcat#1c
\global\advance\colcount by 1\relax\fi
%%
\def\lookforD{#1}
\ifx\lookforD\xD
\global\advance\howmanyDs by 1
\global\advance\colcount by1\relax
\fi
%%
\futurelet\next\lookatnext
}
\makeatletter

\def\xeatone#1{\countcols}
\def\xrelax{\relax}

%% second conditional deals with expressions like @{} in
%% the preamble.
\def\lookatnext{\if\next\xrelax\let\go\relax
\else
\ifx\next\bgroup\let\go\xeatone\else
\let\go\countcols\fi\fi\go}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% reset line counting at \\

%% arraycr redefined to match revtex4-1
\def\new@arraycr{\relax 
\global\columncount=0\relax
\global\colheadcount=0\relax
\iffalse {\fi \ifnum 0=`}\fi \@ifstar {\global \@tbpen \@M \@xarraycr}
{\global \@tbpen \intertabularlinepenalty \@xarraycr}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Redefining & is necessary to get around \omit in \multicolumn,
%% which prevents @{} from being used to change the meaning of multicolumn.
%% Now column number and top or bottom table is used to determine whether
%% multicolumn should be turned on or off.
\let\saveampersand&

\newif\iffirstbox
\newif\ifsecondbox
\newif\ifthirdbox

\long\gdef\CheckNumberAndSwitch{\unskip\global\advance\columncount by 1\relax%
\saveampersand}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parse and count preamble

\def\catchfirstpreamble#1B#2Z{%firstpreamble
\def\leftpreamble{#1\relax}
\colcount=0
\global\howmanyDs=0\relax
\expandafter\countcols\leftpreamble
\global\firstcolcount\the\colcount
\global\firsthowmanyDs\the\howmanyDs
\ifnum\firstcolcount>25
\typeout{^^J
-----------------------------------------------------
^^J
Warning!
^^J^^J 
Too many columns in first part of table!
^^J^^J
Maximum number of columns in each part of the table is 25. Each `D' counts as two columns.
^^J
-----------------------------------------------------
}\fi
\global\howmanyDs=0\relax
\global\colcount=0\relax
\newcolumntype{A}{#1}}

\def\catchsecondpreamble#1B#2Z{%secondpreamble
\def\rightpreamble{#2\relax}
\colcount=0
\expandafter\countcols\rightpreamble
\global\secondcolcount\colcount
\global\secondhowmanyDs\the\howmanyDs
\ifnum\secondcolcount>25
\typeout{^^J-----------------------------------------------------
^^J
Warning!
^^J^^J 
Too many columns in second part of table!
^^J^^J
Maximum number of columns in each part of the table is 25. Each `D' counts as two columns.
^^J
-----------------------------------------------------
}\fi
\global\totalcolumns=\firstcolcount
\global\advance\totalcolumns by \secondcolcount
\global\colcount=0\relax
\newcolumntype{Z}{#2}}


\def\catchsecondofthreepreamble#1B#2B#3Z{%secondpreamble
\def\rightpreamble{#2\relax}
\colcount=0
\howmanyDs=0
\expandafter\countcols\rightpreamble
\global\secondcolcount\colcount
\global\secondhowmanyDs\the\howmanyDs
\global\totalcolumns=\firstcolcount
\global\advance\totalcolumns by \secondcolcount
\global\colcount=0\relax
\newcolumntype{Z}{#2}}


\def\catchthirdpreamble#1B#2B#3Z{%firstpreamble
\def\thirdpreamble{#3}
\colcount=0
\expandafter\countcols\thirdpreamble
\global\thirdcolcount\the\colcount
\global\advance\totalcolumns by \thirdcolcount
\global\colcount=0\relax
\newcolumntype{z}{#3}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define splittabular/endsplittabular

\newcount\PartsOfTable
\newif\ifbreaktab

\def\checkforB #1B#2B#3B{
%% defaults, may be used in \colnumbers
\gdef\tabfullpreamble{#1#2#3\relax}
\gdef\firstpreamble{#1}
\gdef\secondpreamble{#2}
\gdef\thirdpreamble{#3}
\ifx\thirdpreamble\empty
\global\PartsOfTable=2
\else
\global\PartsOfTable=3
\fi
}

\def\lookforD#1{\def\zone{#1}\ifx\zone\xD
\global\Dtrue\else\Dfalse\fi}

\newcount\columnoneandtwo
\newif\ifdonullmulticol

\newcount\colnumsused
\newcount\loopnum

\catcode`\&=\active
\let&\CheckNumberAndSwitch
\let$\savedollar

%sss
\def\splittabular{\catcode`\&=\active
\catcode`\$=\active
\global\colnumbersonfalse
\let\colnumbers\savesplitplaincolnumbers
\let\splitdecimalcolnumbers\savesplitdecimalcolnumbers
\def\decimalcolnumbers{\splitdecimalcolnumbers}
\xsplittabular}

\def\xsplittabular#1#2\end{{
\global\firsttimetrue
\let&\CheckNumberAndSwitch
\let$\savedollar
%%%
\def\endtabular{\endarray
\global\let\zdoit\relax
\global\let\ddoit\relax
\global\Dfalse}
\setbox0=\hbox{\def\firstarg{#1}\expandafter\lookforD\firstarg}
\PartsOfTable=0
\def\tempfullpreamble{#1BB}
\setbox0=\hbox{\expandafter\checkforB\tempfullpreamble}
%% Now PartsOfTable is either =2 or 3
%%
\global\colnumsused=0
\global\breaktabtrue
\global\colcount=0
%%
\ifnum\PartsOfTable=3
\let\multicolumn\threebreakmulticolumn
\else
\let\multicolumn\breakmulticolumn
\fi
%%
\def\one{#1Z}%% full preamble
\ifnum\PartsOfTable=3
\expandafter\catchfirstpreamble\one
\expandafter\catchsecondofthreepreamble\one
\expandafter\catchthirdpreamble\one
\makefirstdummycolumns %f
\makeseconddummycolumns %F
\makethirddummycolumns %E
\else %% PartsOfTable=2
\expandafter\catchfirstpreamble\one
\expandafter\catchsecondpreamble\one
\makefirstdummycolumns %f
\makeseconddummycolumns %F
\fi
%%%
\global\columncount=0
\ifnum\PartsOfTable=3
%% For table divided into three parts !!
\firstboxtrue\secondboxfalse\thirdboxfalse
%
\setbox\firsttablebox=\hbox{%
\begin{tabular}{AFE}% 
#2\crcr
\end{tabular}}
%
\hbox to \hsize{\hss\unhbox\firsttablebox\hss}
\vskip6pt
\hrule
\vskip6pt
\global\columncount=0
\firstboxfalse\secondboxtrue\thirdboxfalse
\ifcolnumberson
\let\colnumbers\xplain
\setbox\secondtablebox=\hbox{%
\begin{tabular}{fZE}% fZE
#2\crcr\end{tabular}}
\else
\setbox\secondtablebox=\hbox{
\begin{tabular}{fZE}% fZE
#2
\end{tabular}}
\fi
\hbox to \hsize{\hss\unhbox\secondtablebox\hss}
%%
\vskip6pt
\hrule
\vskip6pt
\firstboxfalse\secondboxfalse\thirdboxtrue
\global\columncount=0
\ifcolnumberson
\let\colnumbers\xplain
\setbox\thirdtablebox=\hbox{%
\begin{tabular}{fFz}% fFz
#2\crcr\end{tabular}}
\else
\setbox\thirdtablebox=\hbox{
\begin{tabular}{fFz}% fFz
#2
\end{tabular}}
\fi
\hbox to \hsize{\hss\unhbox\thirdtablebox\hss}
\else
%%
%% For table divided into two parts:
%%
\global\columncount=0
\secondboxfalse
 \setbox\firsttablebox=\hbox{%
\begin{tabular}{AF}% should be AF
#2
\end{tabular}
}
 \hbox to \hsize{\hss\unhbox\firsttablebox\hss}
\vskip6pt
\hrule
\vskip6pt
\global\columncount=0
\secondboxtrue
\ifcolnumberson
\setbox\secondtablebox=\hbox{\let\colnumbers\xplain
\begin{tabular}{fZ}% should be fZ
#2
\end{tabular}
}
\else
\setbox\secondtablebox=\hbox{
\begin{tabular}{fZ}% should be fZ
#2
\end{tabular}
}
\fi
\hbox to \hsize{\hss\unhbox\secondtablebox\hss}
%%
%% end of conditional testing for 2 or 3 part table:
\fi
%\global\breaktabfalse
}
\firsthowmanyDs=0
\secondhowmanyDs=0
\resetdecimals
\end
}

%yyy
\def\endsplittabular{\global\firsttimefalse\global\colnumbersonfalse
\global\let\splitplaincolnumbers\savesplitplaincolnumbers
\global\breaktabfalse}

\long\gdef\breakmulticolumn#1#2#3{%
\multispan{#1}%
\let&\CheckNumberAndSwitch
\ifsecondbox%
\ifnum\columncount<\firstcolcount%
\global\donullmulticoltrue%
\else%
\global\donullmulticolfalse%
\fi%
%%%
   \else% first box
\ifnum\columncount<\firstcolcount%
\global\donullmulticolfalse%
\else%
\global\donullmulticoltrue%
\fi%
\fi%
\ifdonullmulticol%
\global\advance\colheadcount by 1\relax
\begingroup
   \def\@addamp{\if@firstamp \@firstampfalse \else
                \@preamerr 5\fi}%
   \@mkpream{h}\@addtopreamble\@empty
   \endgroup
   \def\@sharp{}%
%% comment out \@arstrut to prevent blank line where multicolumn was found
%% on other parts of the table, but not on the current part.
   \@arstrut 
\@preamble
   \null\ignorespaces
\else%
\begingroup
   \def\@addamp{\if@firstamp \@firstampfalse \else
                \@preamerr 5\fi}%
   \@mkpream{#2}\@addtopreamble\@empty
   \endgroup
   \def\@sharp{#3}%
   \@arstrut \@preamble
   \null\ignorespaces
\fi%
\global\donullmulticolfalse%
\global\advance\columncount by #1\relax%
\global\advance\columncount-1\relax%
}


\long\gdef\threebreakmulticolumn#1#2#3{%
\multispan{#1}%
\columnoneandtwo=\firstcolcount\relax%
\advance\columnoneandtwo by \secondcolcount\relax%
%%
\global\donullmulticoltrue\relax%
\iffirstbox\relax%
\ifnum\columncount<\firstcolcount%
\global\donullmulticolfalse%
\fi\relax%
\ifnum\columncount=\firstcolcount\relax%
\donullmulticoltrue\fi\relax%
\else\relax%
%%
\ifsecondbox%
\ifnum\columncount=\firstcolcount\relax%
\global\donullmulticolfalse\fi\relax%
\ifnum\columncount>\firstcolcount%
\ifnum\columncount<\columnoneandtwo\relax%
\global\donullmulticolfalse\relax%
\fi\fi\relax%
\else\relax%
%%%
\ifthirdbox%
\ifnum\columncount<\columnoneandtwo%
\global\donullmulticoltrue%
\else\relax%
\global\donullmulticolfalse%
\fi\relax%
\fi%% end if third box
\fi%% end if second box
\fi%% end if first box
%%
\ifdonullmulticol%
\begingroup\relax%
\global\advance\colheadcount by 1\relax%
   \def\@addamp{\if@firstamp \@firstampfalse \else%
                \@preamerr 5\fi}%
   \@mkpream{h}\@addtopreamble\@empty%
   \endgroup%
   \def\@sharp{}%
%% comment out \@arstrut to prevent blank line where multicolumn was found
%% on other parts of the table, but not on the current part.
   \@arstrut \@preamble%
   \null\ignorespaces
\else%
\begingroup%
   \def\@addamp{\if@firstamp \@firstampfalse \else%
                \@preamerr 5\fi}%
   \@mkpream{#2}\@addtopreamble\@empty%
   \endgroup%
   \def\@sharp{#3}%
   \@arstrut \@preamble%
   \null\ignorespaces
\fi%
\global\donullmulticolfalse%
\global\advance\columncount by #1\relax%
\global\advance\columncount-1\relax%
}

%% First dummy columns is the part of the table that will not print,
%% in this case the first part to the left.
%%\makefirstdummycolumns
%%
%% and second dummy columns
%% \makeseconddummycolumns.
\gdef\makefirstdummycolumns{
\ifcase\firstcolcount
\or% 1
\newcolumntype{f}{h} % no D possible
%%%%%%%%%%%%%%%%%%
\or% 2
  \ifcase\firsthowmanyDs
  \newcolumntype{f}{hh}%% 0
  \or
  \newcolumntype{f}{d} %% 1
  \fi
%%%%%%%%%%%%%%%%%%
\or% 3
  \ifcase\firsthowmanyDs%
  \newcolumntype{f}{hhh}% 0 
  \or%
  \newcolumntype{f}{dh}% 1
  \fi%
%%%%%%%%%%%%%%%%%%
\or% 4
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhh} % 0 
    \or
   \newcolumntype{f}{dhh} % 1
   \or
   \newcolumntype{f}{dd} % 2 
   \fi
%%%%%%%%%%%%%%%%%%
\or% 5
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhh} % 0 
    \or
   \newcolumntype{f}{dhhh} % 1
   \or
   \newcolumntype{f}{ddh} % 2 
   \fi
%%%%%%%%%%%%%%%%%%
\or% 6
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhh} % 1
   \or
   \newcolumntype{f}{ddhh} % 2 
    \or
   \newcolumntype{f}{ddd} % 3
   \fi
%%%%%%%%%%%%%%%
\or % 7
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhh} % 2 
   \or
   \newcolumntype{f}{dddh} % 3
   \fi
%%%%%%%%%%%%%%%%%%
\or % 8
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhh} % 2 
   \or
   \newcolumntype{f}{dddhh} % 3
   \or
   \newcolumntype{f}{dddd} % 4
   \fi
%%%%%%%%%%%%%%%%%%
\or % 9
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhh} % 3
   \or
   \newcolumntype{f}{ddddh} % 4
   \fi
%%%%%%%%%%%%%%%%%%
\or %10
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhh} % 3
   \or
   \newcolumntype{f}{ddddhh} % 4
   \or
   \newcolumntype{f}{ddddd} % 5
   \fi
%%%%%%%%%%%%%%%%%%
\or % 11
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhh} % 4
   \or
   \newcolumntype{f}{dddddh} % 5
   \fi
%%%%%%%%%%%%%%%%%%
\or % 12
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhh} % 4
   \or
   \newcolumntype{f}{dddddhh} % 5
   \or
   \newcolumntype{f}{dddddd} % 6
   \fi
%%%%%%%%%%%%%%%%%%
\or %13
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhh} % 5
   \or
   \newcolumntype{f}{ddddddh} % 6
   \fi
%%%%%%%%%%%%%%%%%%
\or %14
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhhh} % 5
   \or
   \newcolumntype{f}{ddddddhh} % 6
   \or
   \newcolumntype{f}{ddddddd} % 7
   \fi
%%%%%%%%%%%%%%%%%%
\or %15
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhhhh} % 5
   \or
   \newcolumntype{f}{ddddddhhh} % 6
   \or
   \newcolumntype{f}{dddddddh} % 7
   \fi
%%%%%%%%%%%%%%%%%%
\or %16
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhhhhh} % 5
   \or
   \newcolumntype{f}{ddddddhhhh} % 6
   \or
   \newcolumntype{f}{dddddddhh} % 7
   \or
   \newcolumntype{f}{dddddddd} % 8
\fi
%%%%%%%%%%%%%%%%%%
\or %17
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhhhhhh} % 5
   \or
   \newcolumntype{f}{ddddddhhhhh} % 6
   \or
   \newcolumntype{f}{dddddddhhh} % 7
   \or
   \newcolumntype{f}{ddddddddh} % 8
\fi
%%%%%%%%%%%%%%%%%%
\or %18
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhhhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhhhhhhh} % 5
   \or
   \newcolumntype{f}{ddddddhhhhhh} % 6
   \or
   \newcolumntype{f}{ddddddhhhh} % 7
   \or
   \newcolumntype{f}{ddddddddhh} % 8
   \or
   \newcolumntype{f}{ddddddddd} % 9
\fi
%%%%%%%%%%%%%%%%%%
\or %19
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhhhhhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhhhhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhhhhhhhh} % 5
   \or
   \newcolumntype{f}{ddddddhhhhhhh} % 6
   \or
   \newcolumntype{f}{dddddddhhhhh} % 7
   \or
   \newcolumntype{f}{ddddddddhhh} % 8
   \or
   \newcolumntype{f}{ddddddddddh} % 9
\fi
%%%%%%%%%%%%%%%%%%
\or %20
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhh hhhhh hhhhh hhhhh} % 0 
    \or
   \newcolumntype{f}{dhhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhhhhhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhhhhhhhhh} % 5
   \or
   \newcolumntype{f}{ddddddhhhhhhhh} % 6
   \or
   \newcolumntype{f}{dddddddhhhhhh} % 7
   \or
   \newcolumntype{f}{ddddddddhhhh} % 8
   \or
   \newcolumntype{f}{dddddddddhh} % 9
   \or
   \newcolumntype{f}{dddddddddd} % 10
   \fi
%%%%%%%%%%%%%%%%%%
\or %21
  \ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhh hhhhh hhhhh hhhhh h} % 0 
   \or
   \newcolumntype{f}{dhhhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhhhhhhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhhhhhhhhhh} % 5
   \or
   \newcolumntype{f}{ddddddhhhhhhhhh} % 6
   \or
   \newcolumntype{f}{dddddddhhhhhhh} % 7
   \or
   \newcolumntype{f}{ddddddddhhhhh} % 8
   \or
   \newcolumntype{f}{ddd ddd ddd hhh} % 9
   \or
   \newcolumntype{f}{ddd ddd ddd d h} % 10
   \fi
%%%%%%%%%%%%%%%%%%
\or %22
\ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhh hhhhh hhhhh hhhhh hh} % 0 
   \or
   \newcolumntype{f}{dhhhhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhhhhhhhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhhhhhhhhhhh} % 5
   \or
   \newcolumntype{f}{ddddddhhhhhhhhhh} % 6
   \or
   \newcolumntype{f}{dddddddhhhhhhhh} % 7
   \or
   \newcolumntype{f}{ddddddddhhhhhh} % 8
   \or
   \newcolumntype{f}{ddd ddd ddd hhhh} % 9
   \or
   \newcolumntype{f}{ddd ddd ddd dhh} % 10
   \or
   \newcolumntype{f}{ddddddddddd} % 11
   \fi
%%%%%%%%%%%%%%%%%%
\or %23
\ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhh hhhhh hhhhh hhhhh hhh} % 0 
   \or
   \newcolumntype{f}{dhhhhhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhhhhhhhhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhhhhhhhhhhhh} % 5
   \or
   \newcolumntype{f}{ddddddhhhhhhhhhhh} % 6
   \or
   \newcolumntype{f}{dddddddhhhhhhhhh} % 7
   \or
   \newcolumntype{f}{ddddddddhhhhhhh} % 8
   \or
   \newcolumntype{f}{ddd ddd ddd hhhhh} % 9
   \or
   \newcolumntype{f}{ddddddddddhhh} % 10
   \or
   \newcolumntype{f}{ddddd ddddd dh} % 11
   \fi
%%%%%%%%%%%%%%%%%%
\or %24
\ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhh hhhhh hhhhh hhhhh hhhh} % 0 
   \or
   \newcolumntype{f}{dhhhhhhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhhhhhhhhhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhhhhhhhhhhhhh} % 5
   \or
   \newcolumntype{f}{ddddddhhhhhhhhhhhh} % 6
   \or
   \newcolumntype{f}{dddddddhhhhhhhhhh} % 7
   \or
   \newcolumntype{f}{ddddddddhhhhhhhh} % 8
   \or
   \newcolumntype{f}{ddd ddd ddd hhhhhh} % 9
   \or
   \newcolumntype{f}{ddddddddddhhhh} % 10
   \or
   \newcolumntype{f}{ddddd ddddd dhh} % 11
   \or
   \newcolumntype{f}{ddddd ddddd dd} % 12
   \fi
%%%%%%%%%%%%%%%%%%
\or %25
\ifcase\firsthowmanyDs
   \newcolumntype{f}{hhhhh hhhhh hhhhh hhhhh hhhhh} % 0 
   \or
   \newcolumntype{f}{dhhhhhhhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{f}{ddhhhhhhhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{f}{dddhhhhhhhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{f}{ddddhhhhhhhhhhhhhhhhh} % 4
   \or
   \newcolumntype{f}{dddddhhhhhhhhhhhhhhh} % 5
   \or
   \newcolumntype{f}{ddddddhhhhhhhhhhhhh} % 6
   \or
   \newcolumntype{f}{dddddddhhhhhhhhhhh} % 7
   \or
   \newcolumntype{f}{ddddddddhhhhhhhhh} % 8
   \or
   \newcolumntype{f}{ddd ddd ddd hhhhhhh} % 9
   \or
   \newcolumntype{f}{ddddddddddhhhhh} % 10
   \or
   \newcolumntype{f}{dddddddddddhhh} % 11
   \or
   \newcolumntype{f}{ddddddddddddh} % 12
   \fi
%%%%%%%%%%%%%%%%%%
\else
\typeout{^^J----------------------------------------------------- ^^J
Warning!^^J^^J 
Too many Columns using in Splittabular. 
^^J
25 column maximum in
each part of the table.^^J Each`T' counts as two columns.
^^J-----------------------------------------------------^^J}
\fi
}


%% Second dummy columns is the part of the table that will not print,
%% in this case the part to the right.
%%\makeseconddummycolumns

\gdef\makeseconddummycolumns{%
\ifcase\secondcolcount
%0
\or
%1
\newcolumntype{F}{h} % no D possible
%%%%%%%%%%%%%%%%%%
\or
%2
  \ifcase\secondhowmanyDs
  \newcolumntype{F}{hh}%% 0
  \or
  \newcolumntype{F}{d} %% 1
  \fi
%%%%%%%%%%%%%%%%%%
\or
%3
  \ifcase\secondhowmanyDs
  \newcolumntype{F}{hhh} % 0 
  \or
  \newcolumntype{F}{dh} % 1 
  \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhh} % 0 
    \or
   \newcolumntype{F}{dhh} % 1
   \or
   \newcolumntype{F}{dd} % 2 
   \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhh} % 0 
    \or
   \newcolumntype{F}{dhhh} % 1
   \or
   \newcolumntype{F}{ddh} % 2 
   \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhh} % 1
   \or
   \newcolumntype{F}{ddhh} % 2 
   \or
   \newcolumntype{F}{ddd} % 3
   \fi
%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhh} % 2 
   \or
   \newcolumntype{F}{dddh} % 3
   \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhh} % 2 
   \or
   \newcolumntype{F}{dddhh} % 3
   \or
   \newcolumntype{F}{dddd} % 4
   \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhh} % 3
   \or
   \newcolumntype{F}{ddddh} % 4
   \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhh} % 3
   \or
   \newcolumntype{F}{ddddhh} % 4
   \or
   \newcolumntype{F}{ddddd} % 5
   \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhh} % 4
   \or
   \newcolumntype{F}{dddddh} % 5
   \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhh} % 4
   \or
   \newcolumntype{F}{dddddhh} % 5
   \or
   \newcolumntype{F}{dddddd} % 6
   \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhh} % 5
   \or
   \newcolumntype{F}{ddddddh} % 6
   \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhhh} % 5
   \or
   \newcolumntype{F}{ddddddhh} % 6
   \or
   \newcolumntype{F}{ddddddd} % 7
   \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhhhh} % 5
   \or
   \newcolumntype{F}{ddddddhhh} % 6
   \or
   \newcolumntype{F}{dddddddh} % 7
   \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhhhhh} % 5
   \or
   \newcolumntype{F}{ddddddhhhh} % 6
   \or
   \newcolumntype{F}{dddddddhh} % 7
   \or
   \newcolumntype{F}{dddddddd} % 8
\fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhhhhhh} % 5
   \or
   \newcolumntype{F}{ddddddhhhhh} % 6
   \or
   \newcolumntype{F}{dddddddhhh} % 7
   \or
   \newcolumntype{F}{ddddddddh} % 8
\fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhhhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhhhhhhh} % 5
   \or
   \newcolumntype{F}{ddddddhhhhhh} % 6
   \or
   \newcolumntype{F}{dddddddhhhh} % 7
   \or
   \newcolumntype{F}{ddddddddhh} % 8
   \or
   \newcolumntype{F}{ddddddddd} % 9
\fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhhhhhhhhhhhhhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhhhhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhhhhhhhh} % 5
   \or
   \newcolumntype{F}{ddddddhhhhhhh} % 6
   \or
   \newcolumntype{F}{dddddddhhhhh} % 7
   \or
   \newcolumntype{F}{ddddddddhhh} % 8
   \or
   \newcolumntype{F}{dddddddddh} % 9
\fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhh hhhhh hhhhh hhhhh} % 0 
    \or
   \newcolumntype{F}{dhhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhhhhhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhhhhhhhhh} % 5
   \or
   \newcolumntype{F}{ddddddhhhhhhhh} % 6
   \or
   \newcolumntype{F}{dddddddhhhhhh} % 7
   \or
   \newcolumntype{F}{ddddddddhhhh} % 8
   \or
   \newcolumntype{F}{dddddddddhh} % 9
   \or
   \newcolumntype{F}{dddddddddd} % 10
   \fi
%%%%%%%%%%%%%%%%%%
\or
  \ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhh hhhhh hhhhh hhhhh h} % 0 
   \or
   \newcolumntype{F}{dhhhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhhhhhhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhhhhhhhhhh} % 5
   \or
   \newcolumntype{F}{ddddddhhhhhhhhh} % 6
   \or
   \newcolumntype{F}{dddddddhhhhhhh} % 7
   \or
   \newcolumntype{F}{ddddddddhhhhh} % 8
   \or
   \newcolumntype{F}{ddd ddd ddd hhh} % 9
   \or
   \newcolumntype{F}{ddd ddd ddd d h} % 10
   \fi
%%%%%%%%%%%%%%%%%%
\or
\ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhh hhhhh hhhhh hhhhh hh} % 0 
   \or
   \newcolumntype{F}{dhhhhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhhhhhhhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhhhhhhhhhhh} % 5
   \or
   \newcolumntype{F}{ddddddhhhhhhhhhh} % 6
   \or
   \newcolumntype{F}{dddddddhhhhhhhh} % 7
   \or
   \newcolumntype{F}{ddddddddhhhhhh} % 8
   \or
   \newcolumntype{F}{ddd ddd ddd hhhh} % 9
   \or
   \newcolumntype{F}{ddddddddddhh} % 10
   \or
   \newcolumntype{F}{ddddddddddd} % 11
   \fi
%%%%%%%%%%%%%%%%%%
\or
\ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhh hhhhh hhhhh hhhhh hhh} % 0 
   \or
   \newcolumntype{F}{dhhhhhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhhhhhhhhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhhhhhhhhhhhh} % 5
   \or
   \newcolumntype{F}{ddddddhhhhhhhhhhh} % 6
   \or
   \newcolumntype{F}{dddddddhhhhhhhhh} % 7
   \or
   \newcolumntype{F}{ddddddddhhhhhhh} % 8
   \or
   \newcolumntype{F}{ddd ddd ddd hhhhh} % 9
   \or
   \newcolumntype{F}{ddddddddddhhh} % 10
   \or
   \newcolumntype{F}{ddddd ddddd dh} % 11
   \fi
%%%%%%%%%%%%%%%%%%
\or
\ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhh hhhhh hhhhh hhhhh hhhh} % 0 
   \or
   \newcolumntype{F}{dhhhhhhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhhhhhhhhhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhhhhhhhhhhhhh} % 5
   \or
   \newcolumntype{F}{ddddddhhhhhhhhhhhh} % 6
   \or
   \newcolumntype{F}{dddddddhhhhhhhhhh} % 7
   \or
   \newcolumntype{F}{ddddddddhhhhhhhh} % 8
   \or
   \newcolumntype{F}{ddd ddd ddd hhhhhh} % 9
   \or
   \newcolumntype{F}{ddddddddddhhhh} % 10
   \or
   \newcolumntype{F}{ddddd ddddd dhh} % 11
   \or
   \newcolumntype{F}{ddddd ddddd dd} % 12
   \fi
%%%%%%%%%%%%%%%%%%
\or
\ifcase\secondhowmanyDs
   \newcolumntype{F}{hhhhh hhhhh hhhhh hhhhh hhhhh} % 0 
   \or
   \newcolumntype{F}{dhhhhhhhhhhhhhhhhhhhhhhh} % 1
   \or
   \newcolumntype{F}{ddhhhhhhhhhhhhhhhhhhhhh} % 2 
   \or
   \newcolumntype{F}{dddhhhhhhhhhhhhhhhhhhh} % 3
   \or
   \newcolumntype{F}{ddddhhhhhhhhhhhhhhhhh} % 4
   \or
   \newcolumntype{F}{dddddhhhhhhhhhhhhhhh} % 5
   \or
   \newcolumntype{F}{ddddddhhhhhhhhhhhhh} % 6
   \or
   \newcolumntype{F}{dddddddhhhhhhhhhhh} % 7
   \or
   \newcolumntype{F}{ddddddddhhhhhhhhh} % 8
   \or
   \newcolumntype{F}{ddd ddd ddd hhhhhhh} % 9
   \or
   \newcolumntype{F}{ddddddddddhhhhh} % 10
   \or
   \newcolumntype{F}{ddddd ddddd dhhh} % 11
   \or
   \newcolumntype{F}{ddddd ddddd ddh} % 12
   \fi
%%%%%%%%%%%%%%%%%%
\else
\typeout{^^J----------------------------------------------------- ^^J
Warning!^^J^^J 
Too many Columns using in Splittabular. 
^^J
25 column maximum in
each part of the table.^^J Each`D' counts as two columns.
^^J-----------------------------------------------------^^J}
\fi
}

%% we don't have to worry about D's in this section, fortunately.
\gdef\makethirddummycolumns{
\ifcase\thirdcolcount 
\or\newcolumntype{E}{h}
\or\newcolumntype{E}{hh}
\or\newcolumntype{E}{hhh}
\or\newcolumntype{E}{hhhh}
\or\newcolumntype{E}{hhhhh}
\or\newcolumntype{E}{hhhhhh}
\or\newcolumntype{E}{hhhhhhh}
\or\newcolumntype{E}{hhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhhhhhhhhhhhhh}
\or\newcolumntype{E}{hhhhhhhhhhhhhhhhhhhhhhhhh} %% up to 25 columns
\else \typeout{^^J-----------------------------------------------------^^J
Warning!
^^J^^J Too many Columns in Splittabular. 
^^J
25 column maximum in
each part of the table.^^J
Each `D' counts as two columns.
^^J-----------------------------------------------------^^J}\fi
}

%%%%%%%%%%%%%%%%%
%% Colnums

%% defaults

\gdef\settabnumdefaults{%
\loopnum=0\relax\loop%
\ifnum\loopnum<31\relax%
\expandafter\gdef\csname tab\the\loopnum\endcsname{%
\multicolumn1{c}{\global\advance\colnumsused by 1\relax%
(\the\colnumsused)}}%
\global\advance\loopnum by1\repeat}

%%%%%%%%%%%%%%%%%%%%%%%%%
%% this version of countcols  is for \colnumbers
\newcount\totalcolumncount
\def\xH{h}

\makeatother
\def\tabcountcols#1{\ifcat#1c
\global\advance\colcount by 1\relax\fi%
%%
\def\lookforD{#1}%
\ifx\lookforD\xD%
\expandafter\gdef\csname tab\the\colcount\endcsname{%
\multicolumn2{c}{\global\advance\colnumsused by 1\relax%
(\the\colnumsused)}}%
\fi%
\ifx\lookforD\xH%
\expandafter\gdef\csname tab\the\colcount\endcsname{%
\multicolumn1{h}{}}%
\fi%
%%
\futurelet\next\tablookatnext}

\def\tabxeatone#1{\tabcountcols}

%% second conditional deals with expressions like @{} in
%% the preamble.
\def\tablookatnext{\if\next\xrelax\let\xgo\relax\else%
\ifx\next\bgroup\let\xgo\tabxeatone\else\let\xgo\tabcountcols\fi\fi\xgo}
\makeatletter
%%%%%%%%%%%%%%%

\newcount\totalcolumns
\newbox\firsttablebox
\newbox\secondtablebox
\newbox\thirdtablebox
\newif\ifdbreaktab

%% A loop would be more elegant, of course, but using & in a loop
%% within a table column produces errors.
%% was\tabnumberline, now \colnumbers

%% Is this test necessary?
%\newif\ifcolumnums

\newif\ifcolnumberson
\def\deluxecolnumbers{\global\colnumbersontrue}

\def\colnumbers{\omit\\\omit\xcolnumbers}
\def\xcolnumbers{\global\let\colnumbers\xcolnumbers%
\global\colnumbersontrue\let&\CheckNumberAndSwitch%
\global\colcount=0\relax%
\global\totalcolumncount=0\relax%
%% above here
\settabnumdefaults%
%% Now, change the columns that have `D' to be \multicolumn2c, by redefining
%% the \csname tab<number>\endcsname
%%
\expandafter\tabcountcols\tabfullpreamble%
\global\totalcolumncount=\the\colcount\relax%
\global\colcount=0\relax%
\ifbreaktab\else\ifdbreaktab\else\global\colnumsused=0\relax\fi\fi%
%%
%
\ifdeluxe\ifbreaktab\\\else\\[-6pt]\fi%\hline% not above
\else%
\\[-14pt]\fi%
\ifcase\totalcolumncount%
\or%
\csname tab1\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname%
\or
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname%
\or%22
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname%
\or% 24
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname%
\or% 25
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname%
\or% 
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname%
\or% 27
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname&%
\csname tab27\endcsname%
\or% 28
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname&%
\csname tab27\endcsname&\csname tab28\endcsname%
\or% 29
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname&%
\csname tab27\endcsname&\csname tab28\endcsname&%
\csname tab29\endcsname%
\or% 30
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname&%
\csname tab27\endcsname&\csname tab28\endcsname&%
\csname tab29\endcsname&\csname tab30\endcsname%
\else%
\typeout{^^J^^J
Sorry, more than 30 columns cannot be numbered with^^J
\string\colnumbers. Please number the columns manually.^^J
Thank you!^^J}\fi%
%\ifbreaktab
\\\hline% below, hline is wanted
%\else
\noalign{\vskip-8pt}
%\vrule height 28pt width0pt %the \vrule is in the line below colnumbers; 
                          % it causes all the vrules on that line to grow to 28pt.
                          % The -14pt will cause the lower line to overlap the upper line.
                          % Complication, needed to add this vrule to Z and z in order to
                          % have it also work for split tabular.
%\fi
}



\let\savecolnumbers\colnumbers

\gdef\plaincolnumbers{%
\omit\\\omit%
\global\colnumbersontrue\let&\CheckNumberAndSwitch%
\global\colcount=0\relax%
\global\totalcolumncount=0\relax%
%% above here
\settabnumdefaults%
%% Now, change the columns that have `D' to be \multicolumn2c, by redefining
%% the \csname tab<number>\endcsname
%%
\expandafter\tabcountcols\tabfullpreamble%
\global\totalcolumncount=\the\colcount\relax%
\global\colcount=0\relax%
\ifbreaktab\else\ifdbreaktab\else\global\colnumsused=0\relax\fi\fi%
%%
\\\ifcase\totalcolumncount%
\or%
\csname tab1\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname%
\or
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname%
\or%22
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname%
\or% 24
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname%
\or% 25
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname%
\or% 
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname%
\or% 27
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname&%
\csname tab27\endcsname%
\or% 28
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname&%
\csname tab27\endcsname&\csname tab28\endcsname%
\or% 29
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname&%
\csname tab27\endcsname&\csname tab28\endcsname&%
\csname tab29\endcsname%
\or% 30
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname&%
\csname tab27\endcsname&\csname tab28\endcsname&%
\csname tab29\endcsname&\csname tab30\endcsname%
\else%
\typeout{^^J^^J
Sorry, more than 30 columns cannot be numbered with^^J
\string\colnumbers. Please number the columns manually.^^J
Thank you!^^J}\fi%
\\\hline% below, hline is wanted
\\\noalign{\vskip-30pt }%%%%% 
%\vrule height 28pt width0pt %the \vrule is in the line below colnumbers; 
                          % it causes all the vrules on that line to grow to 28pt.
                          % The -14pt will cause the lower line to overlap the upper line.
                          % Complication, needed to add this vrule to Z and z in order to
                          % have it also work for split tabular.
}

\let\saveplaincolnumbers\plaincolnumbers


\def\splitplaincolnumbers{%
\omit\\\omit%
\xplain}

\let\savesplitplaincolnumbers\splitplaincolnumbers

\newif\iffirsttime
\firsttimetrue
\gdef\xplain{%
\global\let\plaincolnumbers\xplain%
\global\colnumbersontrue\let&\CheckNumberAndSwitch%
\global\colcount=0\relax%
\global\totalcolumncount=0\relax%
%% above here
\settabnumdefaults%
%% Now, change the columns that have `D' to be \multicolumn2c, by redefining
%% the \csname tab<number>\endcsname
%%
\expandafter\tabcountcols\tabfullpreamble%
\global\totalcolumncount=\the\colcount\relax%
\global\colcount=0\relax%
\ifbreaktab\else\ifdbreaktab\else\global\colnumsused=0\relax\fi\fi%
%%
\iffirsttime
\\%\hline% above not wanted
\else
\\[-12pt]%\hline% above not wanted
\fi
\ifcase\totalcolumncount%
\or%
\csname tab1\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname%
\or
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname%
\or%22
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname%
\or%
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname%
\or% 24
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname%
\or% 25
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname%
\or% 
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname%
\or% 27
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname&%
\csname tab27\endcsname%
\or% 28
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname&%
\csname tab27\endcsname&\csname tab28\endcsname%
\or% 29
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname&%
\csname tab27\endcsname&\csname tab28\endcsname&%
\csname tab29\endcsname%
\or% 30
\csname tab1\endcsname&\csname tab2\endcsname&%
\csname tab3\endcsname&\csname tab4\endcsname&%
\csname tab5\endcsname&\csname tab6\endcsname&%
\csname tab7\endcsname&\csname tab8\endcsname&%
\csname tab9\endcsname&\csname tab10\endcsname&%
\csname tab11\endcsname&\csname tab12\endcsname&%
\csname tab13\endcsname&\csname tab14\endcsname&%
\csname tab15\endcsname&\csname tab16\endcsname&%
\csname tab17\endcsname&\csname tab18\endcsname&%
\csname tab19\endcsname&\csname tab20\endcsname&%
\csname tab21\endcsname&\csname tab22\endcsname&%
\csname tab23\endcsname&\csname tab24\endcsname&%
\csname tab25\endcsname&\csname tab26\endcsname&%
\csname tab27\endcsname&\csname tab28\endcsname&%
\csname tab29\endcsname&\csname tab30\endcsname%
\else%
\typeout{^^J^^J
Sorry, more than 30 columns cannot be numbered with^^J
\string\colnumbers. Please number the columns manually.^^J
Thank you!^^J}\fi%
\\\hline% below, hline is wanted
\iffirsttime
\\\noalign{\vskip-30pt}%% 
%\noalign{\vskip-8pt}
\vrule height 28pt width0pt %the \vrule is in the line below colnumbers; 
                          % it causes all the vrules on that line to grow to 28pt.
                          % The -14pt will cause the lower line to overlap the upper line.
                          % Complication, needed to add this vrule to Z and z in order to
                          % have it also work for split tabular.
\fi\global\firsttimefalse
}

\def\decimalcolnumbers{\crcr\saveplaincolnumbers\\%
\noalign{\global\savetabskip\tabskip
\tabskip=0pt
\global\let\extracolsep\relax
\global\let\ddoit\saveddoit
\global\let\zdoit\savezdoit
\vskip-32pt 
}{\vrule height28pt depth0pt width0pt}\relax}

\def\splitdecimalcolnumbers{\\[-15pt]%
\savesplitplaincolnumbers
\global\colnumbersonfalse
\\\noalign{\vskip-32pt %!!!
\global\savetabskip\tabskip
\tabskip=0pt
\global\let\extracolsep\relax
\global\let\ddoit\saveddoit
\global\let\zdoit\savezdoit
}\vrule height 28pt width0pt %!!!
\global\let\splitdecimalcolnumbers\secondsplitdecimalcolnumbers}

\def\secondsplitdecimalcolnumbers{\omit\\ 
\plaincolnumbers
\global\colnumbersonfalse
\\\noalign{\vskip-26pt %!!! was -28pt
\global\savetabskip\tabskip
\tabskip=0pt
\global\let\extracolsep\relax
\global\let\ddoit\saveddoit
\global\let\zdoit\savezdoit
}\\\relax}

\let\savesplitdecimalcolnumbers\splitdecimalcolnumbers

%%AAA
\newif\ifdeluxestar
\expandafter\def\csname
deluxetable*\endcsname{\deluxestartrue\bgroup\floattrue
\hsize=\textwidth	
\deluxetable}

\expandafter\def\csname enddeluxetable*\endcsname{\enddeluxetable
\egroup
\null% added june 2020
\global\colnumbersonfalse
\global\deluxedecimalsfalse
\global\deluxestarfalse
}

\def\deluxedecimalcolnumbers{\deluxedecimalstrue\colnumbersontrue}
\def\deluxetablecaption#1{\gdef\@tablecaption{#1}}

\newif\ifstartlongtable
\def\startlongtable{\vskip1sp\global\startlongtabletrue}

\newif\iffloat
\def\floattable{\global\deluxestartrue\global\floattrue}

%% for equivalent but in \begin...\end form.
\def\floatrotatetable{\global\deluxestartrue\global\floattrue}
\let\endfloatrotatetable\relax

\def\deluxetable{\global\deluxetrue
\catcode`\&=\active
\catcode`\$=\active
%% Mar 30, 2019, to make label outside of \caption work correctly
{\advance\c@table by 1
\xdef\@currentlabel{\thetable}}
\let\tablecaption\deluxetablecaption
\deluxetablecaption{}
%% july 2016
% \iffloat
% \let\go\ydeluxetable
% \else\let\go\longdeluxetable\fi\go%}
%% aug 2016
\ifstartlongtable
%% this works for both deluxetable and deluxetable*:
%% nov 2017:
\def\arraystretch{1.1}
%% March 2019 
\if@two@col\global\returntotwocoltrue\vskip1pt
\ifdeluxestar\onecolumngrid\fi\fi
%%
\ifdeluxestar
\vskip12pt
\fi
\let\go\longdeluxetable
\else
\let\go\ydeluxetable
\fi\go
}

\def\ydeluxetable#1{%
\@ifnextchar[{\xdeluxetable{#1}}{\zdeluxetable{#1}}}

%%
% x and z are the same, except that xdeluxetable allows optional square bracket arg, like [h].
\def\xdeluxetable#1[#2]{\global\breaktabtrue
%% not here
\let\colnumbers\deluxecolnumbers
\global\deluxedecimalsfalse
\let\decimals\deluxedecimals
\let\decimalcolnumbers\deluxedecimalcolnumbers
\let\tablehead\ztablehead
\gdef\tabfullpreamble{#1}%<<== needed
\def\endtabular{\endarray
\global\let\zdoit\relax
\global\let\tdoit\relax
\global\Dfalse}
\global\colnumsused=0
\global\breaktabtrue
\global\colcount=0
%%
%%%%%%%%%%%%
  \lineskiplimit=\z@ % restore default setting
\ifdeluxestar
\gdef\two{#2}\gdef\checkh{h}
\ifx\two\checkh\onecolumngrid\fi
\begin{table*}[#2]\hsize=\textwidth\else
  \begin{table}[#2]\fi
\noindent\setbox\splitbox=\vtop\bgroup%
\currtabletypesize
    \vspace*{\abovedeluxetableskip}
%%
\def\startdata##1\enddata{%
\currtabletypesize
\setbox0=\hbox{
\begin{tabular}{#1}
\pt@head
##1\end{tabular}}
\expandafter\ifx\csname @tablecaption\endcsname\empty\else
%% 
\noindent\hbox
to\hsize{\hss\vtop{\hsize=\wd0
\caption{\@tablecaption}}\hss}\vskip3pt\fi

%% not here
\global\setbox3\hbox{%
\begin{tabular}{#1}%
\hline\hline\noalign{\vskip-9pt}%
\pt@head%
##1\crcr% added october 2017
\omit\\\omit\\\hline\end{tabular}}

%\noindent\hskip-1.25em %% kludge! but seems necessary
\hbox to\hsize{\hss\copy3\hss}
\global\setbox4\vtop\bgroup\ifdeluxestar\hsize=\textwidth\else\hsize=\wd0\fi\leftskip6pt\parindent-6pt
}}

\def\zdeluxetable#1{\global\breaktabtrue
\let\colnumbers\deluxecolnumbers
\global\deluxedecimalsfalse
\let\decimals\deluxedecimals
\let\decimalcolnumbers\deluxedecimalcolnumbers
\let\tablehead\ztablehead
\gdef\tabfullpreamble{#1}%<<== needed
\def\endtabular{\endarray
\global\let\zdoit\relax
\global\let\tdoit\relax
\global\Dfalse}
\global\breaktabtrue
\global\colnumsused=0
\global\colcount=0
%%
%%%%%%%%%%%%
  \lineskiplimit=\z@ % restore default setting
\ifdeluxestar
\begin{table*}\hsize=\textwidth\else
  \begin{table}\fi
\noindent\setbox\splitbox=\vtop\bgroup%
\currtabletypesize
    \vspace*{\abovedeluxetableskip}
%%
\def\startdata##1\enddata{%
\currtabletypesize
\setbox0=\hbox{\colnumbersonfalse
\begin{tabular}{#1}
\pt@head %% 
##1\end{tabular}}% not here
\expandafter\ifx\csname @tablecaption\endcsname\empty\else
\noindent\hbox
to\hsize{\hss\vtop{\hsize=\wd0
\caption{\@tablecaption}}\hss}\vskip3pt\fi
\global\setbox3\hbox{%
\begin{tabular}{#1}%
\hline\hline\noalign{\vskip-9pt}%
\pt@head%
##1\crcr%% added October 2017
\omit\\\omit\\\hline\end{tabular}}%%<<== original
\vskip-\parskip
\noindent\hbox to\hsize{\hss\copy3\hss}
\global\setbox4\vtop\bgroup\hsize=\wd0\relax
\leftskip6pt\parindent-6pt
}}


%% go to \end{deluxetable}, after longdeluxetable



%%@@@@

%% Variation on definition found in revtex4-1.cls
\def\LT@start@new{%
 \let\LT@start\endgraf
 \endgraf
 \markthr@@{}%
 \LT@pre
 \@ifvoid\LT@firsthead{\LT@top}{\hbox{\ifdim\movetableright>0pt\relax\hskip\movetableright\fi\box\LT@firsthead}
\nobreak}%
 \mark@envir{longtable}%
}%

%%% 
\newbox\longtablebox
\def\longdeluxetable#1{
\global\rotateonfalse
\let\colnumbers\deluxecolnumbers
\global\deluxedecimalsfalse
\let\decimals\deluxedecimals
\let\decimalcolnumbers\deluxedecimalcolnumbers
\let\tablehead\ztablehead
\gdef\tabfullpreamble{#1}%<<== needed
\def\endtabular{\endarray
\global\let\zdoit\relax
\global\let\tdoit\relax
\global\Dfalse}
\global\colnumsused=0
\global\colcount=0
%%
%%%%%%%%%%%%
  \lineskiplimit=\z@ % restore default setting
\let\enddeluxetable\endlongdeluxetable
    \vspace*{\abovedeluxetableskip}
%%
\def\startdata##1\enddata{%
\global\setbox\longtablebox=\hbox{\currtabletypesize
\tabcolsep=3pt
\begin{tabular}{#1}
\pt@head
##1\end{tabular}}
%%%
\bgroup\centering
\def\table@hook{\currtabletypesize}
      \LTcapwidth=\wd\longtablebox
%% march 2019, added [c] and these terms: 
\ifcenterwidetable\global\centerwidetablefalse
  \def\LT@LR@c{\LTleft=0pt minus1fill 
  \let\LTright\LTleft}%
\else
%% default, will center table that is narrower than text width
  \def\LT@LR@c{\LTleft=0pt plus1fill 
  \LTright\LTleft}%
\fi
%%
%
\begin{longtable}[c]{#1}%%
\ifdim\movetableright>0pt
 \noalign{\ifdim\movetableright>0pt
  \global\LTleft=\movetableright
 \fi}
%
 \noalign{\hbox to \wd\longtablebox{
 \vtop{\hsize=.8\wd\longtablebox 
  \advance\baselineskip4pt
 \raggedright
  {\bf \fnum@table}.\vrule depth 6pt width0pt\
  \@tablecaption}\hss}\vskip-3pt }\\
  \hline
  \hline\noalign{\vskip-9pt}
 \pt@head%
\else
\caption{\hsize=\wd\longtablebox 
\advance\baselineskip2pt
\@tablecaption}\\ %
\hline\hline\noalign{\vskip-9pt}% prob ok
\pt@head%
\fi
\endfirsthead

\noalign{\centerline{\small 
\hskip\movetableright{\bf \fnum@table}\ \it(continued)}\vskip6pt}
\hline\hline
\noalign{\vskip-12pt}
\pt@head%
\endhead

\hline
\multicolumn{\totalcolumns}{c}{\vrule height 24pt width0pt\small\it
\fnum@table\
continued  \if@two@col\else on next page\fi}\\ 
\endfoot

\hline%\nobreak
\endlastfoot
##1
\end{longtable}
\vglue-\LTpost
\vskip-6pt
\egroup
\global\setbox4=\vtop\bgroup
\ifdim\wd\longtablebox>\textwidth
\hsize=\textwidth
\else
\hsize=\wd\longtablebox\fi
\leftskip=6pt
\parindent=-6pt
\currtabletypesize
\global\startlongtablefalse
\global\movetableright=0pt
}% end data, endlongtable
}

\def\endlongdeluxetable{
%%
\vrule depth 6pt width 0pt
\vskip1sp
\egroup
\ifdim\dp4>6pt
\vglue-6pt
\vbox{\hbox to \columnwidth{\hfill
\vtop{\hsize\wd\longtablebox
\leftskip=6pt\parindent-6pt
\copy4
}\hfill}%%
}
\fi
\vglue\ht4 
\global\colnumbersonfalse
\global\deluxedecimalsfalse
\global\rotateonfalse
%%
\relax\null%% \null is an empty hbox.
%% This keeps final page(s) of startlongtable/deluxetable
%% from begin thrown away when at end of article.
% \global\advance\c@table-1\relax
%%
%
%% march 2019
\ifreturntotwocol\global\returntotwocolfalse
\twocolumngrid\fi
%% May 2020
\ifappendixon
\if@two@col@app
\twocolumngrid
\else
\onecolumngrid\fi\fi
\null
}%% 


%% bbb
%% \end{deluxetable}:
\def\enddeluxetable{%
\egroup%end box 4
\ifdim\dp4>6pt
\hbox to\hsize{\hss\copy4\hss}
\fi
\egroup% end splitbox
\centering
\ifrotateon\global\rotateonfalse
\rotatebox{90}{\hbox to\textheight{\hfill\vbox{
\unvbox\splitbox
\vspace*{\belowdeluxetableskip}}\hfill}}
\else
\centering\unvbox\splitbox
\vspace*{\belowdeluxetableskip}
\vspace*{24pt}
\fi
\ifdeluxestar\end{table*}\global\deluxestarfalse
\if@two@col
\twocolumngrid\hsize=\columnwidth\fi%% check this!!
\else
\end{table}\fi
\null
% added june 2020
\gdef\colnumbers{\saveplaincolnumbers}
\global\breaktabfalse
\global\deluxefalse
\global\colnumbersonfalse
\global\deluxedecimalsfalse
\global\rotateonfalse
\startlongtablefalse
\global\movetableright=0pt
%\global\floatfalse
%\global\advance\c@table by -1\relax
%% check this!!
}

%%@@@ ???

\newbox\splitbox
\newif\ifdeluxe
\def\splitdeluxetable{
%% Mar 30, 2019, to make label outside of \caption work correctly
{\advance\c@table by 1
\xdef\@currentlabel{\thetable}}
\global\deluxetrue\catcode`\&=\active
\catcode`\$=\active
\xsplitdeluxetable}

\def\xsplitdeluxetable#1{
\global\breaktabtrue
\let\colnumbers\deluxecolnumbers
\let\decimals\deluxedecimals
\global\deluxedecimalsfalse
\let\decimalcolnumbers\deluxedecimalcolnumbers
\let\tablehead\xtablehead
\gdef\tabfullpreamble{#1}
\def\endtabular{\endarray
\global\let\zdoit\relax
\global\let\tdoit\relax
\global\Dfalse}
%%% modification of emulateapj \deluxetable %%%%
\setbox0=\hbox{\def\firstarg{#1}\expandafter\lookforD\firstarg}
\gdef\temppreamble{#1\relax}
\PartsOfTable=0
\def\tempfullpreamble{#1BB}
\setbox0=\hbox{\expandafter\checkforB\tempfullpreamble}
%% Now PartsOfTable is either =2 or 3
%%
\gdef\one{#1Z}
\global\colnumsused=0
\global\breaktabtrue
\global\colcount=0
%%
\ifnum\PartsOfTable=3
\let\multicolumn\threebreakmulticolumn
\else
\let\multicolumn\breakmulticolumn
\fi
%%%%%%%%%%%%
  \lineskiplimit=\z@ % restore default setting
  \gdef\tblnote@list{}
  \gdef\tblref@list{}
\ifsplitstar
  \begin{table*}\else
  \begin{table}\fi
\noindent\setbox\splitbox=\vtop\bgroup%
\currtabletypesize
    \vspace*{\abovedeluxetableskip}
\let\startdata\xstartdata}
%% \end{splitdeluxetable}:

\newdimen\movetabledown

\def\endsplitdeluxetable{\vskip1sp\egroup%% end box with tablenotes
\hbox to \hsize{\hss\copy4\hss}
\egroup
\vskip1pt
\ifrotateon\global\rotateonfalse
\vskip\movetabledown\rotatebox{90}{\vbox{\noindent\unvbox\splitbox %
\vspace*{\belowdeluxetableskip}
}}\global\movetabledown=0pt\else
\noindent\unvbox\splitbox
\vspace*{\belowdeluxetableskip}
\fi
\ifsplitstar
\end{table*}
\else
\end{table}\fi
\gdef\colnumbers{\saveplaincolnumbers}
\tabletypesize{\small}
\global\breaktabfalse
\global\deluxefalse
\global\splitstarfalse
\global\colnumbersonfalse
\global\deluxedecimalsfalse
}


\newif\ifsplitstar
\expandafter\def\csname splitdeluxetable*\endcsname{\global\splitstartrue\splitdeluxetable}
\expandafter\def\csname endsplitdeluxetable*\endcsname{\endsplitdeluxetable\global\splitstarfalse 
\global\colnumbersonfalse
\global\deluxedecimalsfalse
}

\newdimen\maxtablewidth

\gdef\NoTableCaption{\global\@table@not@headedtrue}

%% For splitdeluxetable

%%% ====================================================== %%%
\gdef\dbreaktabular{\catcode`\&=\active
\let&\CheckNumberAndSwitch\xdbreaktabular}

\long\gdef\xdbreaktabular#1{%% #1=contents of table
{\global\maxtablewidth=0pt
\let&\CheckNumberAndSwitch
\let$\savedollar
\global\colnumsused=0
%%
\global\dbreaktabtrue
\global\colcount=0
%%
\ifnum\PartsOfTable=3
\let\multicolumn\threebreakmulticolumn
\else
\let\multicolumn\breakmulticolumn
\fi
%%
\ifnum\PartsOfTable=3
\expandafter\catchfirstpreamble\one
\expandafter\catchsecondofthreepreamble\one
\expandafter\catchthirdpreamble\one
\makefirstdummycolumns
\makeseconddummycolumns
\makethirddummycolumns
\else %% PartsOfTable=2
\ifnum\PartsOfTable=2
\expandafter\catchfirstpreamble\one
\expandafter\catchsecondpreamble\one
\makefirstdummycolumns
\makeseconddummycolumns
\fi\fi
%%%
\global\columncount=0
\ifnum\PartsOfTable=3
%% For table divided into three parts
\firstboxtrue\secondboxfalse\thirdboxfalse
\setbox\firsttablebox=\hbox{%
\begin{tabular}{AFE}
\ifx\csname pt@head\endcsname\relax\else
\pt@head\fi#1\crcr\end{tabular}}
\global\maxtablewidth=\wd\firsttablebox

\noindent\hbox to \hsize{\hss\unhbox\firsttablebox\hss}
\vskip6pt
\hrule
\vskip6pt
\global\columncount=0
\firstboxfalse\secondboxtrue\thirdboxfalse
\ifcolnumberson%
\setbox\secondtablebox=\hbox{%
\let\savecolnumbers\xcolnumbers%
\notfirsttrue%
\begin{tabular}{fZE} %fZE !!!
\ifx\csname pt@head\endcsname\relax\else%
\pt@head\fi#1\crcr\end{tabular}}
\else
%%
\setbox\secondtablebox=\hbox{%
\begin{tabular}{fZE}%
\ifx\csname pt@head\endcsname\relax\else%
\pt@head\fi\\ [-14pt]% 
#1\crcr%
\end{tabular}}
\fi
\ifdim\wd\secondtablebox>\maxtablewidth
\global\maxtablewidth\wd\secondtablebox\fi

\noindent\hbox to \hsize{\hss\unhbox\secondtablebox\hss}
%%
\vskip6pt
\hrule
\vskip6pt
\firstboxfalse\secondboxfalse\thirdboxtrue
\global\columncount=0
\ifcolnumberson
\notfirsttrue
\setbox\thirdtablebox=\hbox{
\let\savecolnumbers\xcolnumbers
\notfirsttrue
\begin{tabular}{fFz}
\ifx\csname pt@head\endcsname\relax\else
\pt@head\fi#1\crcr\end{tabular}}
\else
\setbox\thirdtablebox=\hbox{
\begin{tabular}{fFz}
\ifx\csname pt@head\endcsname\relax\else
\pt@head\fi\\[-14pt]
#1
\end{tabular}}
\fi
\ifdim\wd\thirdtablebox>\maxtablewidth
\global\maxtablewidth\wd\thirdtablebox\fi

\noindent\hbox to \hsize{\hss\unhbox\thirdtablebox\hss}
\vskip6pt
\hrule
\else
%% Original for table divided into two parts
\global\columncount=0
\secondboxfalse
\setbox\firsttablebox=\hbox{%
\begin{tabular}{AF}
\ifx\csname pt@head\endcsname\relax\else
\pt@head\fi%
#1\end{tabular}}
\global\maxtablewidth=\wd\firsttablebox
\hbox to \hsize{\hss\unhbox\firsttablebox\hss}
\vskip6pt
\hrule
\vskip6pt
\global\columncount=0
\secondboxtrue
\ifcolnumberson
\setbox\secondtablebox=\hbox{%
\let\savecolnumbers\xcolnumbers
\notfirsttrue
\begin{tabular}{fZ}%%
\ifx\csname pt@head\endcsname\relax\else%
\pt@head\fi#1\crcr%%= tablecontents
\end{tabular}}%
\else
\setbox\secondtablebox=\hbox{%
\begin{tabular}{fZ}%% was 12
\ifx\csname pt@head\endcsname\relax\else%
\pt@head\fi\\[-14pt]%
#1%%= tablecontents
\end{tabular}}\fi
\ifdim\wd\secondtablebox>\maxtablewidth
\global\maxtablewidth\wd\secondtablebox\fi

\noindent\hbox to \hsize{\hss\unhbox\secondtablebox\hss}
\vskip6pt
\hrule
\secondboxtrue
%%
%% end of conditional testing for 2 or 3 part table:
\fi
%%
\global\dbreaktabfalse}}
%%% ====================================================== %%%

\newif\ifnotfirst
%% @@@@
%% xstartdata, modified from \startdata in emulateapj, for splitdeluxetable
\gdef\xstartdata#1\enddata{\def\tablecontents{%
\ifcolnumberson%
\\\savecolnumbers\\[2pt]\fi% this is for top level split
\ifdeluxedecimals\savedecimals\fi%
#1}%
\currtabletypesize%
\setbox2=\vtop{\dbreaktabular{\tablecontents}}%
%
\expandafter\ifx\csname @tablecaption\endcsname\empty\else
\noindent\hbox
to\hsize{\hss\vtop{\hsize=\maxtablewidth\caption{\@tablecaption}}\hss}\vskip3pt\fi
\dbreaktabular{\tablecontents\noalign{\global\let\zdoit\relax
\global\let\ddoit\relax}}
\parindent=0pt
\global\setbox4=\vtop\bgroup% egroup in enddeluxetable
\currtabletypesize
\ifsplitstar
\hsize=\textwidth\else\hsize=245.3pt\fi
\parindent=-6pt \leftskip=6pt 
}

\def\tablecomments#1{\vskip1pt{\small\vskip1sp\indent\vrule height 11pt depth 2pt
width 0pt\currtabletypesize{\sc Note}---{#1}\vskip1pt}}

\def\tablenotetext#1#2{\vskip1pt{\currtabletypesize\vskip1pt\indent\vrule
height 11pt depth
2pt width0pt\relax$^{\hbox to 5pt{$#1$}}$#2\vskip1pt}}

\def\tablerefs#1{{\small\vskip3pt\indent\vrule height 11pt depth 2pt
width 0pt\currtabletypesize{\bf References}---{#1}\vskip1sp}}

\let\tablereferences\tablerefs

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. Simple entry decimal numbers

%% extra unused args are to get rid of unnecessary commands revtex puts in
\gdef\lookfordecimal#1#2#3#4.#5 {\gdef\xone{{\let$\relax\savedollar#4\savedollar}}
\gdef\xtwo{#5}\ifx\xtwo\empty\else\gdef\xtwo{{\let$\relax\savedollar#5\savedollar}}\fi}

%% D 
\def\newdoit{\setbox0=\hbox\bgroup\zdoit}
\def\endnewdoit{\egroup\unhbox0}

\long\gdef\zdoit#1 {\let$\relax
\def\bothcols{#1 }%
\setbox0=\hbox{\let$\relax\expandafter\lookfordecimal\bothcols{}{}{}{}.{} }%
\xdoit}

\global\let\savezdoit\zdoit
\let\zdoit\relax

\gdef\xdoit{\relax\hskip\tabcolsep\relax\xone&\tabskip=0pt%
\ifx\xtwo\empty\let\go\relax\else.\def\go{\xtwo}\fi\go}

%% d
\def\xnewdoit{\setbox0=\hbox\bgroup\ddoit}
\def\xendnewdoit{\egroup}

\long\gdef\ddoit#1#2#3{\xxdoit}
\let\saveddoit\ddoit
\let\ddoit\relax

%% to delete all D material: 

\gdef\xxdoit{&}

\newcolumntype{D}{>\newdoit r<{\endnewdoit} @{}l}%% 
\newcolumntype{d}{>\xnewdoit h<{\xendnewdoit} @{}h} %% 

\newdimen\savetabskip
\gdef\decimals{\noalign{\global\savetabskip\tabskip
\tabskip=0pt
\global\let\extracolsep\relax
\global\let\ddoit\saveddoit
\global\let\zdoit\savezdoit
}}

\let\savedecimals\decimals

%% to allow \decimals to be used before \startdata:
\newif\ifdeluxedecimals
\def\deluxedecimals{\global\deluxedecimalstrue}

%%%%%%%%%
%% This needed to be redefined so that \colhead in \tablehead would span both parts
%% of decimal numbers.

%% in case there isn't a table head
\let\pt@head\relax

%% for splitdeluxetable
\def\xtablehead{\catcode`\&=\active%
\catcode`\$=\active%
\xxtablehead}

%%
\def\xxtablehead#1{%
\let&\CheckNumberAndSwitch%
\gdef\pt@head{%
\hline\hline%
\multicolumn1c{\vrule height 12pt width0pt\relax\tempcounter=0
\expandafter\getDs\tabfullpreamble\relax
\global\colcount=0
\global\colheadcount=0}%
\\ 
\iftwelvepoint%
   \ifpreprint%
   \iflongrotateon\else\noalign{\vskip-6.5pt}\fi%
   \else%
      \ifpreprinttwo%
      \iflongrotateon\else\noalign{\vskip-6.5pt}\fi%
      \else%
         \ifmanu%
            \iflongrotateon\noalign{\vskip-8pt}\else\noalign{\vskip-17pt}\fi%
         \else%
           \ifmodern\iflongrotateon\else\noalign{\vskip-9pt}\fi%
            \else%
           \fi%% end ifmodern
         \fi%% end ifmanu
      \fi%% end ifpreprinttwo
    \fi%% end ifpreprint
\else%
%% not twelvepoint
    \iftwocolstyle%
    \iflongrotateon\else\noalign{\vskip-5pt}\fi%
          \else%
              \ifonecolstyle% default
                \iflongrotateon\noalign{\vskip2pt}\else\noalign{\vskip-4.5pt}\fi%
              \fi% end ifonecol
   \fi%endiftwocolstyle
\fi%end iftwelvepoint
#1\ifcolnumberson%
\ifnotfirst\\[-22pt]\fi
\else\\\hline\\[-8pt]\fi% space below hline for 2nd and 3rd part of split table 
}%
%
}

\def\ztablehead{\catcode`\&=\active%
\catcode`\$=\active%
\zztablehead}

\def\zztablehead#1{\let&\CheckNumberAndSwitch%
\gdef\pt@head{%
%% this is needed:
\multicolumn1c{\vrule height12pt width0pt\relax\expandafter\getDs\tabfullpreamble\relax
\global\tempcounter=0
\global\colcount=0
\global\colheadcount=0}\\
\iftwelvepoint%
   \ifpreprint%
   \iflongrotateon\else\noalign{\vskip-6.5pt}\fi%
   \else%
      \ifpreprinttwo%
      \iflongrotateon\else\noalign{\vskip-6.5pt}\fi%
      \else%
         \ifmanu%
            \iflongrotateon\noalign{\vskip-8pt}\else\noalign{\vskip-17pt}\fi%
         \else%
           \ifmodern  \iflongrotateon\else\noalign{\vskip-5pt}\fi%
            \else%
           \fi%% end ifmodern
         \fi%% end ifmanu
      \fi%% end ifpreprinttwo
    \fi%% end ifpreprint
\else%
%% not twelvepoint
    \iftwocolstyle%
    \iflongrotateon\else\noalign{\vskip-5pt}\fi%
          \else%
              \ifonecolstyle% default
                \iflongrotateon\noalign{\vskip2pt}\else\noalign{\vskip-4.5pt}\fi%
              \fi% end ifonecol
   \fi%endiftwocolstyle
\fi%end iftwelvepoint
#1\unskip%
\ifcolnumberson\\[6pt]
\savecolnumbers\vrule height 11pt depth 4pt width 0pt\relax%
\\\ifmanu\noalign{\vskip-15pt}\fi%
\ifdeluxedecimals\\[-14pt]% 
\savedecimals\fi%
\else \vrule depth 6pt
width0pt\\\ifdeluxedecimals\savedecimals\fi\hline\fi%
}}


\newcount\tempcounter

\gdef\getDs#1{\let&\CheckNumberAndSwitch
\def\lookforD{#1}%
\ifx\lookforD\xD%
\expandafter\gdef\csname col\the\tempcounter\endcsname{2}\else
\expandafter\gdef\csname col\the\tempcounter\endcsname{1}\fi 
\global\advance\tempcounter by 1
\futurelet\next\checkingpreamble}

\gdef\checkingpreamble{\ifx\next\relax
\let\go\relax\else\let\go\getDs\fi\go}

%%% Something like this could be used to automate the width of colheads in the tablehead,
%% making those that match the D in preamble are multicolumn2c instead of
%% multicolumn 1c.
\newcount\colheadcount
\newcommand\xcolhead[1]{\multicolumn{\expandafter\ifx\csname
col\the\colheadcount\endcsname\relax 1\else\expandafter%
\csname col\the\colheadcount\endcsname\fi}{c}{\vrule depth 4pt
width0pt\relax#1\relax\global\advance\colheadcount
by 1\relax}\ignorespaces}%
%

%%
\extrarowheight=2pt
%% from revtex4-1, additions on top
%% fix for \usepackage{amsmath} and 
%% any matrix environment
\def\matrixpreamble{*\c@MaxMatrixCols c}
%% this causes problem, must be fixed!
%\let\save@array@array@new\@array@array@new
\def\@array@array@new[#1]#2{%
\ifbreaktab
\else
%%% added so that we can use \colnumbers in any tabular environment
%%% and avoid error messages for breaktabular
\def\catchpreamble{#2\relax}
\gdef\tabfullpreamble{#2\relax}
{\colcount=0
%% important fix, makes it possible to use amsmath and matrices.
\def\two{#2}
\ifx\matrixpreamble\two
\else
\expandafter\countcols\catchpreamble
\global\totalcolumns=\colcount\fi}
\fi
%%% end of additions
  \@tempdima\ht\strutbox
  \advance\@tempdima by\extrarowheight
  \setbox\@arstrutbox\hbox{%
   \vrule \@height\arraystretch\@tempdima
          \@depth \arraystretch\dp\strutbox
          \@width \z@
  }%
  \begingroup
   \@mkpream{#2}%
   \xdef\@preamble{\@preamble}%
  \endgroup
  \prepdef\@preamble{%
   \tabskip\tabmid@skip
    \@arstrut
  }%
  \appdef\@preamble{%
   \tabskip\tabright@skip
   \cr
   \array@row@pre
  }%
  \@arrayleft
  \@nameuse{@array@align@#1}%
  \m@th
  \let\\\new@arraycr %<<== defined above, original plus reset counters
  \let\tabularnewline\\%
  \let\par\@empty
  \let\@sharp##%
  \set@typeset@protect
  \lineskip\z@\baselineskip\z@
  \tabskip\tableft@skip
  \everycr{}%
  \expandafter\halign\expandafter\@halignto\expandafter\bgroup\@preamble
}%

%% \rotate 
\newif\ifrotateon
\def\rotate{\global\floattrue\iffloat\global\let\go\relax\global\rotateontrue\else
\global\rotateonfalse 
\let\go\relax
%\def\go{\vskip48pt\huge
%{\tt \string\floattable\space must precede table using
%\string\rotate !!}}
\fi\go}

\def\colnumbers{\plaincolnumbers}

%% add 6pt of space between table and following text:
\let\saveendtable\endtable
\def\endtable{\vskip6pt\saveendtable}
\expandafter \let \csname endtable*\endcsname = \endtable
%%


%%% \movetabledown works
\newbox\rotatetablebox
\def\rotatetable{%
\clearpage
\global\startlongtabletrue\setbox\rotatetablebox=\vbox\bgroup
}

\def\endrotatetable{\egroup
\vglue\movetabledown
\hbox to
\textwidth{\hss\rotatebox{90}{\hbox{\box\rotatetablebox}}\hss}
\global\movetabledown=0pt\relax
\global\startlongtablefalse
}

%%% \movetabledown works
\expandafter\def\csname rotatetable*\endcsname{%
\clearpage
\global\startlongtabletrue\setbox\rotatetablebox=\vbox to
\textwidth\bgroup\vfill}

\expandafter\def\csname endrotatetable*\endcsname{\vfill\egroup
\vbox to \textheight{\vfill
\vglue\movetabledown
\hbox to
\textwidth{\hss\rotatebox{90}{\box\rotatetablebox}\hss}\global\movetabledown=0pt\relax 
\vfill}
\clearpage
\if@two@col
\twocolumngrid
\global\startlongtablefalse
}

%
%%% \movetabledown works
\newif\iflongrotateon
\def\longrotatetable{%
\global\longrotateontrue
   \if@two@col\onecolumngrid\clearpage
   \fi
\clearpage
   \clearpage
  \begingroup
\expandafter\def\csname deluxetable*\endcsname{\deluxetable}
\expandafter\def\csname enddeluxetable*\endcsname{\enddeluxetable}
\global\floatfalse
\global\startlongtabletrue
\def\LS@rot{%
  \setbox\@outputbox\vbox{\vskip\movetabledown\hbox{\rotatebox{90}{\box\@outputbox}}}}

\let\LS@makecol=\@makecol
\let\LS@makefcolumn=\@makefcolumn

  \vsize=\textwidth
  \hsize=\textheight
  \linewidth=\hsize
  \columnwidth=\hsize
  \@colroom=\vsize
  \textheight=\vsize
  \@colht=\vsize
  \def\@makecol{\LS@makecol\LS@rot}%
  \def\@makefcolumn##1{\LS@makefcolumn{##1}\LS@rot}}

\def\endlongrotatetable{%
\onecolumngrid %% ??
\clearpage
  \ifGin@pdftex
    \pdfpageattr{/Rotate 90}
  \fi
  \clearpage
  \endgroup
   \if@two@col\twocolumngrid\fi
\clearpage
\global\longrotateonfalse
\global\movetabledown=0pt
  \global\@colht=\textheight
  \global\vsize=\textheight
  \global\@colroom=\textheight}

\newif\ifGin@pdftex
\Gin@pdftexfalse
\DeclareOption{pdftex}{%
  \PassOptionsToPackage\CurrentOption{graphicx}
  \Gin@pdftextrue
}

\DeclareOption*{\PassOptionsToPackage\CurrentOption{graphicx}}
\ProcessOptions

%% set &, $ catcode back to normal:

\catcode`&=4
\catcode`$=3



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% end table tools
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Gridlines for positioning multiple illustrations in 
%% one figure environment.

\def\gridline#1{\vskip6pt\hbox to\hsize{#1}\vskip6pt}

\def\boxedfig#1#2#3{\hfill\fbox{\vbox{\parskip=0pt\relax
\hsize=#2
\includegraphics[width=#2]{#1}\vskip2pt\vtop{\hsize=#2
\centerline{#3}}}}\hfill}

\def\fig#1#2#3{\hfill\vbox{\parskip=0pt\hsize=#2
\includegraphics[width=#2]{#1}\vskip2pt\vtop{\centering
\footnotesize
\hsize=#2
#3\vskip1pt
}}\hfill}

\def\leftfig#1#2#3{\vbox{\parskip=0pt\relax\hsize=#2
\includegraphics[width=#2]{#1}\vskip2pt\vtop{\hsize=#2
\centering
#3\vskip1sp\vskip1sp}}\hfill}

\def\rightfig#1#2#3{\hfill\vbox{\parskip=0pt\relax\hsize=#2
\includegraphics[width=#2]{#1}\vskip2pt\vtop{\hsize=#2
\centering#3\vskip1sp}}}

\def\rotatefig#1#2#3#4{\hfill\vbox{\centering\parskip=0pt\hsize=#3
\includegraphics[width=#3,angle=#1]{#2}\vskip2pt\vtop{\centering
\footnotesize
\hsize=#3
#4\vskip1pt
}}\hfill}

%%%%%%%%%% End Grid line Macros %%%%%%%%%%

%%%%%%%%%% Color Editing Macros %%%%%%%%%%
%% \turnoffedit or \turnoffedits
%% will prevent all \edit<number>{text}, all
%% \collaborationcomment<number>{text}, and all
%% \authorcomment<number>{text} from producing any text.

\newif\ifturnoffedit
\def\turnoffedit{\global\turnoffedittrue}
\def\turnoffedits{\global\turnoffedittrue}
\let\turnoffediting\turnoffedits

\newif\ifturnoffone
\newif\ifturnofftwo
\newif\ifturnoffthree

%% \turnoffeditone to only turn off only \edit1, \collaborationcomment1 and \authorcomment1 :
\def\turnoffeditone{\turnoffonetrue}


%% \turnoffedittwo to only turn off only \edit2, \collaborationcomment2 and \authorcomment2 :
\def\turnoffedittwo{\turnofftwotrue}


%% \turnoffeditthree to only turn off only \edit3, \collaborationcomment3 and \authorcomment3 :
\def\turnoffeditthree{\turnoffthreetrue}


%% Other choices can be made, but this should be
%% standardized, so didn't make an user interface
%% to change the colors easily.

\expandafter\def\csname editcolor1\endcsname{black}% was magenta
\expandafter\def\csname editcolor2\endcsname{black}% was blue
\expandafter\def\csname editcolor3\endcsname{black}% was violet

\let\newgo\relax
\newcount\colorcount
	
\newcount\editnum
\def\edit#1#2{\ifcase#1\or\ifturnoffone%
\unskip%
\else%
\texorpdfstring{{\bf#2}}{#2}\fi\or%
\ifturnofftwo%
\unskip%
\else\texorpdfstring{{\bfseries\itshape#2}}{#2}\fi%
\or%
\ifturnoffthree\unskip\else%
\texorpdfstring{{\bfseries\underline{#2}}}{#2}\fi\fi}

\def\collaborationcomment#1#2{{\colorcount=#1
\ifturnoffedit\let\go\ignorespaces\else%
\let\go\relax%
\color{\csname editcolor\the\colorcount\endcsname}
\ifnum\colorcount=1{\ifturnoffone\else\bf (Collaboration~note: #2)\ \fi}\else
\ifnum\colorcount=2{\ifturnofftwo\else\it (Collaboration~note: #2)\ \fi}\else
\ifnum\colorcount=3{\ifturnoffthree\else (Collaboration~note: #2)\ \fi}\fi\fi\fi\fi%
\go}}

\def\authorcomment#1#2{{\colorcount=#1
\ifturnoffedit\let\go\ignorespaces\else%
\let\go\relax%
\color{\csname editcolor\the\colorcount\endcsname}
\ifnum\colorcount=1{\ifturnoffone\else\bf (Author~note 1: #2)\ \fi}\else
\ifnum\colorcount=2{\ifturnofftwo\else\it (Author~note 2: #2)\ \fi}\else
\ifnum\colorcount=3{\ifturnoffthree\else (Author~note 3: #2)\ \fi}\fi\fi\fi\fi%
\go}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Draft watermark

\definecolor{ltgray}{gray}{.9}% .89
\font\bighelv=phvr at 2in %
\def\setwatermarkfontsize#1{\font\bighelv=phvr at #1}

\def\watermark#1{\def\thewatermark{\hbox to\textwidth{\vtop to
1.1\textheight{\vss
\hskip24pt\rotatebox{60}{\hbox{\bighelv \color{ltgray} 
\uppercase{#1}}}\vss}}\hss}}
\let\thewatermark\empty

%% Used in titlepage definition, as \pagestyle{titlepage}
\def\ps@titlepage{%
  \let\@mkboth\@gobbletwo
\def\@oddhead{\ifx\thewatermark\empty\hfill\else
\hbox to \textwidth{\rlap{\thewatermark}\hfill}\fi}
\let\@oddfoot\@empty
\let\@evenhead\@empty\let\@evenfoot\@empty}

% end of watermark definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand*\submitjournal[1]
{\def\@submitted{Submitted to #1}}%

%% this one is used &&&
\def\titleblock@produce{%
 \begingroup
 \ltx@footnote@pop
 \def\@mpfn{mpfootnote}% 
 \def\thempfn{\thempfootnote}%
 \c@mpfootnote\z@
 \let\@makefnmark\frontmatter@makefnmark
  \frontmatter@setup
  \thispagestyle{titlepage}\label{FirstPage}%
\ifmodern\leftskip=0pt\rightskip\leftskip\fi
  \frontmatter@title@produce
  \groupauthors@sw{%
\frontmatter@author@produce@group
  }{%
   \frontmatter@author@produce@script
  }%
  \frontmatter@RRAPformat{%
\vskip12pt
   \expandafter\produce@RRAP\expandafter{\@date}%
   \expandafter\produce@RRAP\expandafter{\@received}%
   \expandafter\produce@RRAP\expandafter{\@revised}%
   \expandafter\produce@RRAP\expandafter{\@accepted}%
   \expandafter\produce@RRAP\expandafter{\@published}%
  }%
\expandafter\ifx\csname @submitted\endcsname\relax\else
\vskip6pt
\expandafter\produce@RRAP\expandafter{\centerline{\@submitted\hbox
to 20pt{\hfill}}\vskip12pt}%
\fi
  \frontmatter@abstract@produce
  \@ifx@empty\@pacs{}{%
   \@pacs@produce\@pacs
  }%
  \@ifx@empty\@keywords{}{%
   \@keywords@produce\@keywords
  }%
  \par
  \frontmatter@finalspace
\endgroup%
}%


% needed??
\gdef\specialbibitem#1[#2]#3#4{
{\tt\string\bibitem}[#2]{\tt\string{\string}}
#4}



%%%%%%%%%%%%%%%%%%%%%%%%%
%% Track Changes
%% Amy Hendrickson, Nov 2015
%% Change Jan 2016, to allow list of changes
%% to give line and page numbers for more than one entry on the same line.
%% Change Feb 2016, to allow optional argument for time/date, and/or editor initials, etc.


\providecolor{trackchange}{cmyk}{0,0,0,1}
\providecolor{explain}{cmyk}{0,0,0,1}

\newif\ifsilent

\newcount\refchangenumber
\def\added{\@ifnextchar[{\xadded}{\yadded}}

\long\def\xadded[#1]#2{%
\iftrack {\global\advance\refchangenumber by 1\relax%
\vtop to 0pt{\vss
\hypertarget{link\the\refchangenumber}{}
\vskip14pt}%
\ifnumlines%
\ifabstract\else%
\xdef\doit{\noexpand\linelabel{\the\refchangenumber}}\doit\fi\else%
\xdef\doit{\noexpand\label{\the\refchangenumber}{}{}{}}\doit\fi}%
{\color{trackchange}\bf(Added: [#1] #2)}%%
\ifabstract\label{\the\refchangenumber}%
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{\bf Added: [#1]
\textcolor{trackchange}\bf\relax{#2}\global\silenttrue}%
\else\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{\bf\relax Added: [#1]
\textcolor{trackchange}\bf\relax{\let\bibitem\specialbibitem #2}\global\silentfalse}\fi%
\else#2\fi}



\long\def\yadded#1{%
\iftrack{\global\advance\refchangenumber by 1\relax%
\vtop to 0pt{\vss
\hypertarget{link\the\refchangenumber}{}
\vskip14pt}%
\ifnumlines\ifabstract\else%
\xdef\doit{\noexpand\linelabel{\the\refchangenumber}}\doit\fi\else%
\xdef\doit{\noexpand\label{\the\refchangenumber}{}{}{}}\doit%
\fi}%
{\color{trackchange}\bf(Added: #1)}%%
\ifabstract%
\label{\the\refchangenumber}%
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{\bf Added:
\textcolor{trackchange}\bf\relax{#1},
\global\silenttrue}\else
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{\bf Added:
\textcolor{trackchange}{\bf\relax\let\bibitem\specialbibitem #1},%
\global\silentfalse}\fi%
\else#1\fi}



\def\deleted{\@ifnextchar[{\xdeleted}{\ydeleted}}

\long\def\xdeleted[#1]#2{\iftrack{\global\advance\refchangenumber by
1\relax%
\vtop to 0pt{\vss
\hypertarget{link\the\refchangenumber}{}
\vskip14pt}%
\ifnumlines\ifabstract\else%
\xdef\doit{\noexpand\linelabel{\the\refchangenumber}}\doit\fi%
\else%
\xdef\doit{\noexpand\label{\the\refchangenumber}}\doit\fi%
}%
{\color{trackchange}\bf%
\ifbib\let\sout\relax\fi%
(Deleted: [#1] \sout{#2})
}%
\ifabstract\label{\the\refchangenumber}%
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{Deleted: [#1]
{\color{trackchange}\bf%
\sout{#2}}\global\silenttrue}%
\else
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{Deleted: [#1] 
{\color{trackchange}%
\let\bibitem\specialbibitem%
\sout{#2}
}\global\silentfalse}\fi%
\fi}

\long\def\ydeleted#1{\iftrack{\global\advance\refchangenumber by 1\relax%
\vtop to 0pt{\vss
\hypertarget{link\the\refchangenumber}{}
\vskip14pt}%
\ifnumlines\ifabstract\else%
\xdef\doit{\noexpand\linelabel{\the\refchangenumber}}\doit\fi%
\else%
\xdef\doit{\noexpand\label{\the\refchangenumber}{}{}{}}\doit%
\fi}%
%%
{\color{trackchange}\bf%
\ifbib\let\sout\relax\fi%
(Deleted: \sout{#1})}%
\ifabstract\label{\the\refchangenumber}%
\expandafter\gdef\csname changenum\the\refchangenumber\endcsname{Deleted:
{\color{trackchange}\bf
\sout{#1}}\global\silenttrue}%
\else
\expandafter\gdef\csname changenum\the\refchangenumber\endcsname{Deleted:
{\color{trackchange}\bf%
\sout{#1}}\global\silentfalse}\fi%
\fi}


\def\replaced{\@ifnextchar[{\xreplaced}{\yreplaced}}

\long\def\xreplaced[#1]#2#3{%
\iftrack\global\advance\refchangenumber by 1\relax%
\vtop to 0pt{\vss
\hypertarget{link\the\refchangenumber}{}
\vskip14pt}%
\ifnumlines\ifabstract\else\xdef\doit{\noexpand\linelabel{\the\refchangenumber}}\doit\fi\else%
\xdef\doit{\noexpand\label{\the\refchangenumber}}\doit\fi%
{\ifbib\let\sout\relax\fi
\color{trackchange}\bf(Replaced: [#1] \sout{#2}}%
{\color{black}replaced with:} {
\color{trackchange}\bf\relax #3)}%
\ifabstract\label{\the\refchangenumber}%
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{Replaced: [#1]
{\color{trackchange}\bf\relax\sout{#2}} {\color{black} replaced with:}
{\color{trackchange}\bf\relax#3}, \global\silenttrue}%
\else
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{Replaced: [#1]
{\ifbib\let\sout\relax\fi
\color{trackchange}\bf\relax\sout{#2}
} {\color{black} replaced with:}
{\color{trackchange}\bf\relax#3}, \global\silentfalse}\fi%
\else#3\fi}

\long\def\yreplaced#1#2{%
\iftrack\global\advance\refchangenumber by 1\relax%
\vtop to 0pt{\vss
\hypertarget{link\the\refchangenumber}{}
\vskip14pt}%
\ifnumlines\ifabstract\else\xdef\doit{\noexpand\linelabel{\the\refchangenumber}}\doit\fi\else%
\xdef\doit{\noexpand\label{\the\refchangenumber}}\doit\fi%
{\ifbib\let\sout\relax\fi
\color{trackchange}\bf\relax(Replaced: 
\sout{#1}
}%
{\color{black}replaced with:}
{\color{trackchange}\bf\relax #2)}%
\ifabstract\label{\the\refchangenumber}%
\expandafter\gdef\csname changenum\the\refchangenumber\endcsname{Replaced:
{\color{trackchange}\bf\relax
\sout{#1}} {\color{black} replaced with:}
{\color{trackchange}\bf\relax#2},\global\silenttrue}%
\else
\expandafter\gdef\csname
changenum\the\refchangenumber\endcsname{Replaced:
{\ifbib\let\sout\relax\fi
\color{trackchange}\bf\relax\sout{#1}
} {\color{black} replaced with:}
{\let\bibitem\specialbibitem\color{trackchange}\bf\relax#2}, \global\silentfalse}\fi%
\else#2\fi}

\def\explain{\@ifnextchar[{\xexplain}{\yexplain}}

\def\xexplain[#1]#2{\iftrack\ {\bfseries\itshape\color{explain} [Explanation of change:
#2 (#1)]\ }\fi}

\def\yexplain#1{\iftrack\ {\bfseries\itshape\color{explain} [Explanation of change:
#1]\ }\fi}


\newcount\listchangenum

\def\listofchanges{
\clearpage
\iftrack
\ifnum\refchangenumber>0
\ifnumlines\nolinenumbers\fi
\vskip36pt
\hrule
\noindent{\vrule height 14pt width0pt depth 6pt\large\bf List of Changes}
\hrule
\vskip18pt
\nobreak
{\parskip=4pt \parindent=0pt
\loop
\global\silentfalse
\vskip-1pt\relax
\global\advance\listchangenum by 1\relax
\expandafter\ifx\csname changenum\the\listchangenum\endcsname\relax
\else
\csname changenum\the\listchangenum\endcsname\ 
on page
%% can't get hyperlink to work correctly for page numbers, works but error messages.
\hyperlink{link\the\listchangenum}{\bf\pageref{\the\listchangenum}}%
\ifnumlines%
\ifsilent\global\silentfalse \else%
\setbox0=\hbox{\lineref{\the\listchangenum}}%
\ifdim\wd0<20pt%
, line\
\hyperlink{link\the\listchangenum}{\lineref{\the\listchangenum}}%
\fi
\fi\fi.\relax
\repeat}
\fi\fi
\thispagestyle{empty}
}


\PassOptionsToPackage{normalem}{ulem}
\usepackage{ulem}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% From earlier version of AASTeX, for usefulness and 
%% backward compatibility, with some requested additions

\let\jnl@style=\rmfamily
\def\ref@jnl#1{{\jnl@style#1}}%

%%%
\renewcommand\memsai{\ref@jnl{Mem.~Soc.~Astron.~Italiana}}%
          % Mem. Societa Astronomica Italiana
%% longer version:
\newcommand\memsocai{Mem. Societ\`a Astronomica Italiana}
\newcommand\aspconf{Ast. Soc. of the Pac. Conference Series}

\let\astap=\aap
\let\apjlett=\apjl
\let\apjsupp=\apjs
\let\applopt=\ao

%%% More useful commands from Earlier version of Aastex:
\let\la=\lesssim            % For Springer A&A compliance...
\let\ga=\gtrsim

\let\farcm\farcm@mss
\let\farcs\farcs@mss

\def\farcm@apj{%
 \mbox{.\kern -0.7ex\raisebox{.9ex}{\scriptsize$\prime$}}%
}%

\def\farcs@apj{%
 \mbox{%
  \kern  0.13ex.%
  \kern -0.95ex\raisebox{.9ex}{\scriptsize$\prime\prime$}%
  \kern -0.1ex%
 }%
}%

\def\chem@bnd#1{%
 {%
  \kern.1em\relax
  \setbox\z@\hbox{M}%
  \dimen@ii.8em\relax
  \p@=.1em\relax
  \dimen@.5\ht\z@\dimen@i-\dimen@
  \advance\dimen@1.5\p@\advance\dimen@i-1.0\p@
  #1%
  \kern.1em\relax
  }%
 }%
\def\@sbnd{%
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 }%
\def\@dbnd{%
 \advance\dimen@-0.5\p@\advance\dimen@i0.5\p@
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \hskip-\dimen@ii
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 }%
\def\@tbnd{%
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \hskip-\dimen@ii
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 \advance\dimen@-1.5\p@\advance\dimen@i1.5\p@
 \hskip-\dimen@ii
 \vrule\@height\dimen@\@depth\dimen@i\@width\dimen@ii\nobreak
 }%

\renewcommand\LaTeX{%
 \leavevmode
 L%
 \raise.42ex\hbox{%
  \count@=\the\fam
  $\fam\count@\scriptstyle\kern-.3em A$%
 }%
 \kern-.15em\TeX
}%

%% added Feb 2, 2016, redefining definition of \caption made with hyperref
%% package. Making \caption[] be the default, to prevent sending caption 
%% text to listoftables or listoffigures, which we are not going to use
%% anyway. This change enables track changes commands to work in captions.
\def\xtable{table}
\def\caption{\numlinesfalse
\ifx\@captype\@undefined 
\@latex@error {\noexpand \caption outside float}\@ehd 
\expandafter \@gobble \else 
\H@refstepcounter \@captype \let \Hy@tempa \@caption \@ifundefined
{float@caption}{}
{\expandafter \ifx \csname @float@c@\@captype\endcsname
\float@caption\let\Hy@tempa\Hy@float@caption\fi} 
\expandafter\@firstofone\fi 
{\@dblarg {\Hy@tempa \@captype}}[]}

\def\@caption#1[#2]#3{{\small\rm\expandafter \ifx \csname if@capstart\expandafter \endcsname 
\csname iftrue\endcsname \global \let \@currentHref \hc@currentHref \else \hyper@makecurrent {\@captype }\fi 
\@ifundefined {NR@gettitle}{\def \@currentlabelname{\thesection\relax #2}}
{\NR@gettitle {#2}}\par 
%\addcontentsline {\csname ext@#1\endcsname }{#1}{\protect \numberline {\csname the#1\endcsname }{\ignorespaces #2}}
\begingroup \@parboxrestore \if@minipage \@setminipage \fi \normalsize 
\expandafter\ifx \csname if@capstart\expandafter \endcsname 
\csname iftrue\endcsname \global \@capstartfalse 
\@makecaption {\bf\csname fnum@#1\endcsname }{\ignorespaces #3}
\else \@makecaption {\bf\csname fnum@#1\endcsname }{\ignorespaces \ifHy@nesting 
\expandafter \hyper@@anchor \expandafter {\@currentHref }{#3}\else 
\Hy@raisedlink {\expandafter \hyper@@anchor \expandafter
{\@currentHref }{\relax }}{#3}\fi }\fi \par
\endgroup}}

%% changed, March 23, 2019, took out \\ at end of definition:
\newcommand\tablebreak{\\[-11pt]\noalign{\break}}

%% As suggested by Greg Schwarz, August Meunch, Feb 11

\def\dataset{\def\doi##1{https://doi.org/##1}
\@ifnextchar[{\ydataset}{\xdataset}}
\def\xdataset#1{\ydataset[]{#1}\let\doi\savedoi}
\def\ydataset[#1]#2{\def\one{#1}\ifx\one\empty
\href{#2}{[DATASET]}\else
\href{#2}{#1}\fi\let\doi\savedoi}

%% this seems in the middle of nowhere...
\advance\columnsep6pt

%%%%%%%%%%
%% macros to give hyperref link to authors when orcid numbers
%% are supplied with \author[0000-2345-3333-0023]{author name}
 \def\lookforbracket{\ifx\next[\let\go\xauthor\else\let\go\yauthor\fi\go}

\newcount\entriesinthiscollab 
\newcount\allentries
\newif\ifseesmessage
\def\author{\global\advance\entriesinthiscollab by 1\relax%
\global\advance\allentries by 1\relax\futurelet\next\lookforbracket}

\def\new@author@def#1#2{%
 \move@AU\move@AF\move@AUAF
 \let\@AF@join\@author@join
 \def\@author{{\href{http://orcid.org/#1}{#2% 
\openin1 Orcid-ID.png \ifeof1
%% message for authors??
%\typeout{^^J^^J  ! Missing File: Orcid-ID.png; needed for Orcid Author icon !
%^^J}
\else%
\hskip2pt\includegraphics[width=9pt]{Orcid-ID.png}\fi}}{}}%
}%

\def\orciderrormessage{%
\typeout{^^J^^J [\firstarg]\space Invalid ORCID Identifier!^^J^^J The ID
should consist of four sets of four digits,^^J separated with -, ie,
0000-0012-3245-1234 or ^^J
0000-0012-3245-123X
^^J^^J
Please fix, or don't use square bracket argument with
\string\author[]\string{\string}
^^J^^J}%
}

%% Tests to see if author gave reasonable Orcid number;
%% if not, stops processing and gives error message.
\newif\ifbadOrcid
\def\lookatOrcid #1-#2-#3-#4-{%
\def\one{#1}\def\two{#2}\def\three{#3}\def\four{#4}%
\ifx\one\empty%
\global\badOrcidtrue%
\else%
\ifx\two\empty%
\global\badOrcidtrue%
\else%
\ifx\three\empty%
\global\badOrcidtrue%
\else%
\ifx\four\empty%
\global\badOrcidtrue%
\fi\fi\fi\fi}

\def\lookforDigits #1#2#3#4{
\def\xone{#1}\def\xtwo{#2}\def\xthree{#3}\def\xfour{#4}
%%
\ifx\xone\empty 
\global\badOrcidtrue
\else
  \ifcat \xone 1\else
  \global\badOrcidtrue\fi
\fi
%%%
\ifx\xtwo\empty 
\global\badOrcidtrue
\else
 \ifcat \xtwo 1\else
  \global\badOrcidtrue\fi
\fi
%%%
\ifx\xthree\empty 
\global\badOrcidtrue
\else
 \ifcat \xthree 1\else
  \global\badOrcidtrue\fi
\fi
%%%
\ifx\xfour\empty 
\global\badOrcidtrue
\else
 \ifcat \xfour 1\else
  \global\badOrcidtrue\fi
\fi}

\def\FinallookforDigits #1#2#3#4{
\def\xone{#1}\def\xtwo{#2}\def\xthree{#3}\def\xfour{#4}
%%
\ifx\xone\empty 
\global\badOrcidtrue
\else
  \ifcat \xone 1\else
  \global\badOrcidtrue\fi
\fi
%%%
\ifx\xtwo\empty 
\global\badOrcidtrue
\else
 \ifcat \xtwo 1\else
  \global\badOrcidtrue\fi
\fi
%%%
\ifx\xthree\empty 
\global\badOrcidtrue
\else
 \ifcat \xthree 1\else
  \global\badOrcidtrue\fi
\fi
%%%
\ifx\xfour\empty 
\global\badOrcidtrue
\else
   \ifcat \xfour 1\else
      \if\xfour X
      \else
      \global\badOrcidtrue\fi%
   \fi%
\fi}





%% Last digit might be X
\def\xauthor[#1]#2{\def\testone{#1-}\def\firstarg{#1}%
\setbox0=\hbox{\expandafter\lookatOrcid\testone {}-{}-{}-{}-
\expandafter\lookforDigits\one {}{}{}{}%
\expandafter\lookforDigits\two {}{}{}{}%
\expandafter\lookforDigits\three {}{}{}{}%
\expandafter\FinallookforDigits\four {}{}{}{}}%
%%
\ifbadOrcid\let\go\orciderrormessage
\else\let\go\relax\fi\go%
\ifbadOrcid\let\go\stop\else\let\go\relax\fi\go%
\new@author@def{#1}{#2}}%% oct 7, this doesn't seem to work

\def\yauthor{\@author@def{}}

\def\altaffilmark#1{\typeout{^^J^^J\string\altaffilmark\space is no longer
needed. It will not do anything.^^J Please use
\string\altaffiliation\string{\string} instead.^^J^^J}\stop}

\def\altaffiltext#1{\typeout{^^J^^J\string\altaffiltext\space is no longer
needed. It will not do anything.^^J Please use
\string\altaffiliation\string{\string} instead.^^J^^J}\stop}

%% We do not need \and in list of authors. It is taken care of
%% automatically. Below is the definition from revtex4-1:
% 
%\renewcommand\frontmatter@and{\class@err{\protect\and\space is not
% supported}}

\newcount\AuthorCollaborationLimit
\let\AuthorCallLimit\AuthorCollaborationLimit
%% no limit for default
\AuthorCollaborationLimit=10000
\newcount\largestAffilNum

\def\lookfornumbers#1#2#3#4#5#6#7#8#9{\def\one{#1}
\def\two{#2}
\def\three{#3}
\def\four{#4}
\def\five{#5}
\def\six{#6}
\def\seven{#7}
\def\eight{#8}
\def\nine{#9}
\ifnum\one>\largestAffilNum
\global\largestAffilNum=\one\fi
%%%
\ifx\two\empty\else
\ifnum\two>\largestAffilNum
\global\largestAffilNum=\two\fi\fi
%%%
\ifx\three\empty\else
\ifnum\three>\largestAffilNum
\global\largestAffilNum=\three\fi\fi
%%%
\ifx\four\empty\else
\ifnum\four>\largestAffilNum
\global\largestAffilNum=\four\fi\fi
%%%
\ifx\five\empty\else
\ifnum\five>\largestAffilNum
\global\largestAffilNum=\five\fi\fi
%%%
\ifx\six\empty\else
\ifnum\six>\largestAffilNum
\global\largestAffilNum=\six\fi\fi
%%%
\ifx\seven\empty\else
\ifnum\seven>\largestAffilNum
\global\largestAffilNum=\seven\fi\fi
%%%
\ifx\eight\empty\else
\ifnum\eight>\largestAffilNum
\global\largestAffilNum=\eight\fi\fi
%%%
\ifx\nine\empty\else
\ifnum\nine>\largestAffilNum
\global\largestAffilNum=\nine\fi\fi
}

%\gdef\newcomma@space{\hskip-3pt\textsuperscript{,}}%
\def\doEtAl{\rm et al.\gdef\doEtAl{\relax}}
%% =====================
\newif\iffirsttime
\firsttimetrue
\newcount\totalentries
\newcount\docollabnum
\newcount\tempauthornumber
\newcount\countauthors
\newif\ifdothis
\def\doAnd{}
\newcount\testnum

\def\olddoauthor#1#2#3{%
\iflongauthor\vskip6pt\fi
\global\advance\countauthors by 1
\ifnum\countauthors>\AuthorCollaborationLimit
\doEtAl
\else
\ifnum\AuthorCollaborationLimit=1
\let\@listand\relax
\fi
  \ignorespaces\leavevmode\hbox{#1\unskip\@listcomma}% nice, keeps name from breaking across lines
\fi
  \begingroup
\ifnum\countauthors>\AuthorCollaborationLimit\else
  #3% all affil numbers
\ifx\@affilID@temp\empty
\else%
\setbox0=\hbox{\expandafter\lookfornumbers\@affilID@temp{}{}{}{}{}{}{}{}{}}%
\fi\fi
%% #2= \altaffiliation{} or \email{} or 
%% possibly anything other than author, affiliation, or collaboration 
  \@if@empty{#2}{\endgroup{}{}}
{\ifnum\countauthors>\AuthorCollaborationLimit\endgroup{}{}%% <<< bug fix, added \endgroup{}{}
\else
\endgroup{\comma@space}{}\frontmatter@footnote{#2}\fi}%
\ifnum\countauthors>\AuthorCollaborationLimit\else  \space
\@listand\fi 
}%

\newif\ifaddspace
\def\doauthor#1#2#3{%
\iflongauthor\vskip6pt\fi%%%
\ifanonymous%
    \iffirsttime%
     \global\firsttimefalse%
     Anonymous author(s)%
    \fi%
\else%% ends at end of this def
%%%
\ifnum\docollabnum< 1
\global\AuthorCollaborationLimit\expandafter\csname
currCollabLimit0\endcsname%
%% this won't change until after collaboration name at end
\fi %% end ifnum\docollabnum
%%
%%%%%%%%%%%%%%%%
\global\advance\totalentries by 1\relax%
\global\advance\countauthors by 1\relax%
%%%
\ifallauthors\global\AuthorCollaborationLimit=9999 \fi%
%%%
% for testing
% [author number=\the\countauthors/ auth collab limit
% =\the\AuthorCollaborationLimit]
%
\gdef\xone{#1}%
\ifnum\countauthors < \AuthorCollaborationLimit
\global\addspacetrue
\gdef\docomma{,}\else\gdef\docomma{}\fi% ok
%%%
\ifnum\countauthors = \AuthorCollaborationLimit
\ifnum\AuthorCollaborationLimit=1\else
\gdef\doAnd{And }\fi%
\ifnum\tempauthornumber= 1
\gdef\doAnd{  } \fi\fi%
%%
{\tempauthorminusone=\AuthorCollaborationLimit
\advance\tempauthorminusone by -1
\ifnum\countauthors=\tempauthorminusone
\gdef\doAnd{And }%
\gdef\docomma{}\fi}%
%%%
\global\dothisfalse%
%% if num countauthors is less than or equal to \AuthorCollaborationLimit, print name
\ifnum\countauthors< \AuthorCollaborationLimit
%%% 
\ifx\xone\empty\else%
\global\dothistrue%
%\expandafter\gdef\csname
%dothisaffil-\the\countauthors\the\docollab\endcsname{dothisone}
  \ignorespaces\leavevmode\hbox{#1\unskip\docomma}% nice, keeps name from breaking across lines
\fi%
\fi%
%%
\ifnum\countauthors= \AuthorCollaborationLimit
%%% 
\ifx\xone\empty\else
\global\dothistrue
  \ignorespaces\leavevmode\hbox{\doAnd #1\unskip\docomma}%nice, keeps name from breaking across lines
\fi%
\fi%
%% ++++
%%
\ifsuppressAffiliations\else%
\ifx\xone\empty\else%
\begingroup%
\ifnum\countauthors>\AuthorCollaborationLimit\else
  #3% all affil numbers
\ifx\@affilID@temp\empty %% number following author
\else%
\setbox0=\hbox{\expandafter\lookfornumbers\@affilID@temp{}{}{}{}{}{}{}{}{}}%
\fi\fi%
%% #2= \altaffiliation{} or \email{} or 
%% possibly anything other than author, affiliation, or collaboration 
  \@if@empty{#2}{\endgroup{}{}}%
{\ifnum\countauthors>\AuthorCollaborationLimit\endgroup{}{}%% <<< bug fix, added \endgroup{}{}
\else
\endgroup{\comma@space}{}\frontmatter@footnote{#2}\fi}%
\fi%% end test of empty
\fi%% end test of suppressAffiliations
%%%%%%%%%%%%%%%%%%%%%%%%
%%% Collaboration name is always used if available:
\expandafter\ifx\csname currCollabName\the\totalentries\endcsname\relax\else 
%%%
%%%
%%%
%% changed locally:
{\advance\docollabnum 1 
\advance\countauthors-1
%%%
%%% We need to test to see if more than one author in collaboration
%%% and if the number of authors is greater than the \AuthorCollaborationLimit.
%%% If both of these are true, than use `et al.' , otherwise do not:
\expandafter\ifx\csname
CollabTotalAuthors\the\docollabnum\endcsname\relax
\else
\expandafter\ifnum\csname
CollabTotalAuthors\the\docollabnum\endcsname
< 2 %%%%% Don't use et al if there is only one author
\else
\expandafter\ifnum\csname CollabTotalAuthors\the\docollabnum\endcsname
>\AuthorCollaborationLimit\relax%
\ifnum\AuthorCollaborationLimit=0\else
\vskip3pt
{\rm et al.} \vskip-3pt
\fi\fi\fi\fi%
}%% end local change to docollabnum
%%%%
\expandafter\ifx\csname
currCollabName\the\totalentries\endcsname\empty
\vskip-3pt
\else
\vskip6pt
\expandafter\csname currCollabName\the\totalentries\endcsname\vskip8pt
\affiliation{testing}
\fi%
%%%%%%%%
\global\countauthors=0
%%%
%%%
\global\advance\docollabnum  by 1
%% Set up counters for next time through this loop:
{\advance\docollabnum by1
  \expandafter\ifx\csname
  specificCollabLimit\the\docollabnum\endcsname\relax
 \else
   \global\AuthorCollaborationLimit=\csname
   specificCollabLimit\the\docollabnum\endcsname 
  \fi%
\expandafter\ifx\csname
CollabTotalAuthors\the\docollabnum\endcsname\relax
\else
\global\tempauthornumber=\csname
CollabTotalAuthors\the\docollabnum\endcsname
\fi}%
\fi %% ends test to see if it is time to use collaboration name
\fi %% ends ifanonymous
\ifaddspace\ \fi\global\addspacefalse}% oct 7, 2020, allows multiple names to break over lines,
                                         %but no extra space if \collaboration{1}


%%%% ++++====
\newcount\tempauthorminusone
%% +++
\def\doAllauthors#1#2#3{%
\global\suppressAffiliationsfalse
\iflongauthor\vskip6pt\fi
%%%
\ifanonymous
    \iffirsttime
     \global\firsttimefalse
     Anonymous author(s)
    \fi
\else %% ends at end of this def
%%%
\ifnum\docollabnum< 1
%% these won't change until after collaboration name at end
\global\AuthorCollaborationLimit\expandafter\csname currCollabLimit0\endcsname 
\global\tempauthornumber=\csname CollabTotalAuthors1\endcsname
\fi%% end ifnum\docollabnum
%%
%%%%%%%%%%%%%%%%
\global\advance\totalentries by 1
\global\advance\countauthors by 1
%%%
\ifallauthors\global\AuthorCollaborationLimit=9999 \fi
%%%
% for testing
%[author number=\the\countauthors/ temp author
%=\the\tempauthornumber]
\def\one{#1}
{\tempauthorminusone=\tempauthornumber
\advance\tempauthorminusone by -1
\ifnum\countauthors < \tempauthornumber
\gdef\xdocomma{,}%
\else\gdef\xdocomma{}\fi%
%%%
\ifnum\countauthors = \tempauthornumber
\gdef\xdoAnd{And}\gdef\xdocomma{}%
\else\gdef\xdoAnd{}\fi%
\ifnum\tempauthornumber= 1
\gdef\xdoAnd{}\gdef\xdocomma{}\fi%
\ifnum\countauthors=\tempauthorminusone
\gdef\xdocomma{}\fi%
}
%%%
%% if num countauthors is less than or equal to \AuthorCollaborationLimit, print name
\ifnum\countauthors< \AuthorCollaborationLimit
%%% 
\ifx\one\empty\else
  \ignorespaces\leavevmode\hbox{\unskip\xdoAnd\ #1\unskip\xdocomma}% nice, keeps name from breaking across lines
\fi\fi%
%%
\ifnum\countauthors= \AuthorCollaborationLimit
%%% 
\ifx\one\empty\else
\ignorespaces\leavevmode\hbox{\unskip\xdoAnd\ #1\unskip\xdocomma}% nice, keeps name from breaking across lines
\fi%
\fi%
%%
\ifx\one\empty\else
 \begingroup
\ifnum\countauthors>\AuthorCollaborationLimit\else
  #3% all affil numbers
\ifx\@affilID@temp\empty
\else%
\setbox0=\hbox{\expandafter\lookfornumbers\@affilID@temp{}{}{}{}{}{}{}{}{}}%
\fi\fi
%% #2= \altaffiliation{} or \email{} or 
%% possibly anything other than author, affiliation, or collaboration 
  \@if@empty{#2}{\endgroup{}{}}
{\ifnum\countauthors>\AuthorCollaborationLimit\endgroup{}{}%% <<< bug fix, added \endgroup{}{}
\else
\endgroup{\comma@space}{}\frontmatter@footnote{#2}\fi}%
\fi%% end test of empty
%%%%%%%%%%%%%%%%%%%%%%%%
%%% Collaboration name is always used if available:
\expandafter\ifx\csname currCollabName\the\totalentries\endcsname\relax\else 
%%%
%% No et al in allauthors, because all authors are listed!
%%
\expandafter\ifx\csname
currCollabName\the\totalentries\endcsname\empty
\else
\vskip6pt
\expandafter\csname currCollabName\the\totalentries\endcsname\vskip6pt
\fi
\global\countauthors=0
%%%
%%%
\global\advance\docollabnum  by 1
%% Set up counters for next time through this loop:
{\advance\docollabnum by1
  \expandafter\ifx\csname
  specificCollabLimit\the\docollabnum\endcsname\relax
 \else
   \global\AuthorCollaborationLimit=\csname
   specificCollabLimit\the\docollabnum\endcsname 
  \fi
%
\expandafter\ifx\csname
CollabTotalAuthors\the\docollabnum\endcsname\relax
\else
\global\tempauthornumber=\csname
CollabTotalAuthors\the\docollabnum\endcsname
\fi}
\fi %% ends test to see if it is time to use collaboration name
\fi %% ends ifanonymous
}% 

%%%
%% set \affil to match \affiliation found in revtex, since authors are accustomed to using \affil{}
\let\affil\affiliation

\newif\iffirstaffil
\firstaffiltrue

\newcount\affilnum
%% +++!
\def\@affil@script#1#2#3#4{%
\ifsuppressAffiliations\else
\iffirstaffil
% Oct 2017
\vskip2pt
\global\firstaffilfalse\fi
 \@ifnum{#1=\z@}{}{%
  \par
  \begingroup
   \frontmatter@affiliationfont
   \@ifnum{\c@affil<\affil@cutoff}{}{%
\def\one{#1}
%%
%%
\ifnum\one<\largestAffilNum
%% this makes the numbers
   \def\@thefnmark{#1}\@makefnmark\fi
\ifnum\one=\largestAffilNum
   \def\@thefnmark{#1}\@makefnmark\fi
   }%
\ifnum\one<\largestAffilNum
   \ignorespaces#3%
\fi
\ifnum\one=\largestAffilNum
   \ignorespaces#3%
\fi
   \@if@empty{#4}{}{\frontmatter@footnote{#4}}%
   \par
  \endgroup
 }%
\fi}%



\newif\ifnobreakafterkeywords
\def\NoNewPageAfterKeywords{\global\nobreakafterkeywordstrue}
\NoNewPageAfterKeywords
\def\NewPageAfterKeywords{\global\nobreakafterkeywordsfalse}

%%%+++!!!
\def\frontmatter@maketitle{%
%% \@author@finish needs to be here:
\@author@finish
\if@firstsection% not appendix, in other words
  \title@column\titleblock@produce 
\onecolumngrid
\else
%
\global\firstaffiltrue
 \title@column\secondtitleblock@produce
\fi
  \suppressfloats[t]%
%% Prevent these from being turned off so that
%% we can use \maketitle again for \AllAuthors.
%%
%  \let\and\relax
  \let\affiliation\@gobble
  \let\author\@gobble
%  \let\@AAC@list\@empty
%  \let\@AFF@list\@empty
%  \let\@AFG@list\@empty
%  \let\@AF@join\@AF@join@error
%  \let\email\@gobble
%  \let\@address\@empty
% \let\maketitle\relax
%  \let\thanks\@gobble
\if@firstsection
\let\abstract\@undefined\let\endabstract\@undefined
\titlepage@sw{%
\ifnobreakafterkeywords
\else
\vfil
\ifrnaas\else \clearpage\fi
\fi
 }{}%
\fi
}%


\let\maketitle\frontmatter@maketitle

\def\secondtitleblock@produce{%
 \begingroup
  \ltx@footnote@pop
  \def\@mpfn{footnote}%
  \def\thempfn{\thefootnote}%
  \c@footnote\z@
  \let\@makefnmark\frontmatter@makefnmark
%  \frontmatter@setup
  \thispagestyle{titlepage}%\label{FirstPage}%
%  \frontmatter@title@produce
  \groupauthors@sw{%
   \frontmatter@author@produce@group
  }{%
   \frontmatter@author@produce@script
  }%
  \endgroup
}%

%% no club or widow lines
\widowpenalty=10000
\clubpenalty\widowpenalty
\setlength{\footnotesep}{8pt}

\ifmodern
\setlength{\voffset}{0in}
\setlength{\hoffset}{0in}
\setlength{\textwidth}{6in}
\setlength{\textheight}{9.2in}
\setlength{\headheight}{0ex}
\setlength{\headsep}{36pt} % this is 2 lines in ``manuscript''
\setlength{\footnotesep}{0in}
\setlength{\topmargin}{-\headsep}
\setlength{\oddsidemargin}{0.25in}
\setlength{\evensidemargin}{0.25in}
\setlength{\parindent}{0.54\baselineskip}
\sloppy\sloppypar
\fi


\def\figurename{Figure}
\def\tablename{Table}
\def\fnum@figure{{\bf\figurename~\thefigure}}
\def\fnum@table{{\bf\tablename~\ifappletter\thesection\fi\thetable}}


\def\tempfootmark#1{}

\newcount\c@affilcount
\renewcommand*\altaffiliation[2][]{%
\@AF@join{\ifanonymous\else
#1#2\ifmodern\baselineskip=14pt\fi
\if@two@col\hsize=.5\textwidth
\advance\hsize by -18pt
\fi\fi}%
}%


\def\correspondingauthor#1{{\ifanonymous
\else
\renewcommand\thefootnote{\hskip-12pt}
\footnote{Corresponding author: #1\ifmodern\vrule depth 5pt
width 0pt\relax\fi}\fi}}

\let\saveemail\email
\def\email#1{\ifanonymous
\else{\let\ltx@footmark\tempfootmark
\saveemail{}}
{\renewcommand\thefootnote{\hskip-12.1pt}
\footnote{\href{mailto: #1}{#1}\ifmodern\vrule depth 7pt width
0pt\relax\else\ifmanu\vskip-4pt\else\vrule depth 7pt width
0pt\fi\fi}}\fi}

\def\nocollaboration#1{%
\collaboration{#1}{\vbox to
0pt{\vss\centerline{---}\vskip2pt}}
}

%% May 19
%% \AuthorsAndCollaboration changed to \FullCollaborationID 
%% June 6 \FullCollaborationID changed to \xcollaboration{}{}
%% june 7 \collaboration changed to \xcollaboration; fullcollaborationid changed to \collaboration

%% here just in case we need it in the future...
\def\xcollaboration#1#2{
\global\advance\allentries by 1
\expandafter\def\csname
currCollabLimit\the\allentries\endcsname{#1}
\@author@def{\@booleantrue
\collaboration@sw}{#2}
}

\let\savelistand\@listand
\newcount\numauthors
\newcount\collabnum
\newbox\collabnamebox
\newif\iffirstcollab
\global\firstcollabtrue

\newif\ifcollaborationon
\def\collaboration#1#2{\global\collaborationontrue
\global\advance\collabnum by 1
\iffirstcollab\global\firstcollabfalse
\expandafter\xdef\csname currCollabLimit0\endcsname{#1}
\fi
%% make this def so that we can use it when we want in doauthor:
\expandafter\gdef\csname specificCollabLimit\the\collabnum\endcsname{#1} 
%% This allows us access the number of authors per collaboration:
\expandafter\xdef\csname
CollabTotalAuthors\the\collabnum\endcsname{\the\entriesinthiscollab} 
\global\entriesinthiscollab=0
%%
\global\advance\allentries by 1
%% 
%% we should have only one of these with this number; this used to say when
%% use specificCollabLimit:
\expandafter\gdef\csname currCollabLimit\the\allentries\endcsname{#1}
%% we should have only one of these with this number
\expandafter\gdef\csname
currCollabName\the\allentries\endcsname{%\ifnum#1>0
%\sc And the\vskip4pt\fi
#2}
%%
\let\doEtAl\relax
%% not here
\@author@def{\@booleanfalse
\collaboration@sw}{}}


\def\and{
\centerline{\vbox {\vrule height 12pt width0pt and\vskip2pt}}
}

%\def\andthe{%
%\collaboration{\vbox {\vrule height 12pt width0pt and
%the\vskip2pt}}\vskip4pt
%}

%% gets rid of () around collaboration
\def\@collaboration@present#1#2#3#4{%
\par
 \begingroup
\vskip3pt
\iflongauthor\vskip-4pt\ifmodern\vskip-6pt\fi\fi
  \frontmatter@collaboration@above
  \@affilID@def{}%
  \@tempcnta\z@
  \@author@present{}{\ignorespaces#3\unskip}{#4}%
%% October 2017
  \par
\ifmodern%\vskip-4pt
\else
\iflongauthor\else\vskip-6pt\fi\fi
 \endgroup
\iflongauthor 
\else\vskip8pt\fi
 \set@listcomma@list#1%
\vskip1pt %% was \vskip9pt
}%


%%% These lines were commented out until a fix could be applied that
%%% addresses the underlying issues. The problem is that on Linux systems
%%% you can not write a "hidden" .bib file. There is no issue with this
%%% on Mac OS X nor Windows.
%%% get rid of \jobname Notes being sent to .aux file:
%\let\bibdata@app\relax
%\def\pre@bibdata{}

%% This is the underlying issue:
%% Redefining \@bibdataout@init will keep useless file, \jobnameNotes, from being written.
\let\@bibdataout@init\relax

%% redefining and simplifying RevTeX4-1 definition, so that bibliography is not looking for
%% \jobnameNotes:

\def\bibliography#1{\bibliography@latex{#1}}%


%%%
\newif\iffrontmatterfirsttime
\global\frontmatterfirsttimetrue
\let\savecentercr\@centercr
\def\frontmatter@author@produce@script{%
  \begingroup
    \let\@author@present\@author@present@script
    \frontmatterverbose@sw{\typeout{\string\frontmatter@author@produce@script:}\say\@AAC@list\say\@AFF@list\say\@AFG@list}{}%
    \let\AU@temp\@empty
    \@tempcnta\z@
    \let\AF@opr \@gobble
    \def\AU@opr{\@author@count\@tempcnta}%
    \def\CO@opr{\@collaboration@count\AU@temp\@tempcnta}%
    \@AAC@list
    \expandafter\CO@opr\@author@cleared
    \begingroup
     \frontmatter@authorformat
     \let\AF@opr \@affilID@def
     \let\AU@opr \@author@present
     \def\CO@opr{\@collaboration@present\AU@temp}%
     \set@listcomma@list\AU@temp
     \@AAC@list
%% \, added to definition taken from revtex4-1 to prevent error message.
\,     \unskip\unskip
     \par
    \endgroup
%% below
    \begingroup
     \frontmatter@above@affiliation@script
     \let\AFF@opr \@affil@script
{\parskip=2pt
\def\@centercr{\vrule depth 3pt width0pt\vskip1sp}
     \@AFF@list}
\let\@centercr\savecentercr
%% sept 14, 2020
\iffrontmatterfirsttime
\frontmatter@footnote@produce
\global\frontmatterfirsttimefalse
\fi
     \par
    \endgroup
  \endgroup
}%

\let\auto@bib\relax

\newif\iffirstaffil
\firstaffiltrue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% These commands were in aastex earlier; I redefined.
\def\tighten{\global\tightentrue\normalsize}
\let\tightenlines=\tighten
\def\singlespace{\par\global\doublespacefalse\global\tightenfalse\normalsize}
\def\doublespace{\par\global\doublespacetrue\global\tightenfalse\normalsize}

\bibliographystyle{aasjournal}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% October 2017

\renewenvironment{widetext}{%
\if@two@col
  \par\ignorespaces
  \setbox\widetext@top\vbox{%
   \hb@xt@\hsize{%
    \leaders\hrule\hfil
    \vrule\@height6\p@
   }%
  }%
  \setbox\widetext@bot\hb@xt@\hsize{%
    \vrule\@depth6\p@
    \leaders\hrule\hfil
  }%
  \onecolumngrid
  \vskip10\p@
  \dimen@\ht\widetext@top\advance\dimen@\dp\widetext@top
  \cleaders\box\widetext@top\vskip\dimen@
  \vskip6\p@
  \prep@math@patch
\fi}{%
\if@two@col
  \par
  \vskip6\p@
  \setbox\widetext@bot\vbox{%
   \hb@xt@\hsize{\hfil\box\widetext@bot}%
  }%
  \dimen@\ht\widetext@bot\advance\dimen@\dp\widetext@bot
  \cleaders\box\widetext@bot\vskip\dimen@
  \vskip8.5\p@
  \twocolumngrid\global\@ignoretrue
  \@endpetrue
\fi}%
\newbox\widetext@top
\newbox\widetext@bot

\urlstyle{rm}

\def\doi#1{\href{https://doi.org/#1}{https://doi.org/#1}}
\let\savedoi\doi

\def\mdash{---}

\newif\ifcenterwidetable
%% these two definitions are the same, but it might
%% be convenient to have both
\def\centerwidetable{\global\centerwidetabletrue}


%% this doesn't seem to be documented, assume we don't need it:
%\def\centernarrowtable{\global\centerwidetablefalse}

%% default, center within width of text on both sides
 \def\LT@LR@c{\LTleft=0pt plus1fill 
  \LTright\LTleft}%

\def\widetable{\def\LT@LR@c{\LTleft=0pt minus1fill 
  \let\LTright\LTleft}}

\newif\ifreturntotwocol


%%%%%%%%%
%% to help with graceful linebreaks in two column text:
  \tolerance 9999%
%% sloppy defines emergencystretch to be 3 em, this is a bit
%% more conservative:
  \emergencystretch 1em 
  \hfuzz .5\p@

%%%%%%%%%
%% Nominal Conversion Constants

%%  \boldmath inside the \hbox ??
\def\nomSolarEffTemp{\leavevmode\hbox{\boldmath$\mathcal{T}^{\rm N}_{\mathrm{eff}\odot}$}}
\def\nomTerrEqRadius{\leavevmode\hbox{\boldmath$\mathcal{R}^{\rm N}_{E\mathrm e}$}}
\def\nomTerrPolarRadius{\leavevmode\hbox{\boldmath$\mathcal{R}^{\rm N}_{E\mathrm p}$}}
\def\nomJovianEqRadius{\leavevmode\hbox{\boldmath$\mathcal{R}^{\rm
N}_{J\mathrm e}$}}
 \def\nomJovianPolarRadius{\leavevmode\hbox{\boldmath$\mathcal{R}^{\rm
 N}_{J\mathrm p}$}}
 \def\nomTerrMass{\leavevmode\hbox{\boldmath$(\mathcal{GM})^{\rm N}_{\mathrm E}$}}
 \def\nomJovianMass{\leavevmode\hbox{\boldmath$(\mathcal{GM})^{\rm N}_{\mathrm J}$}}
 \def\Qnom{\leavevmode\hbox{\boldmath$\mathcal{Q}^{\rm N}_{\odot}$}}
\let\Qn\Qnom

%% Generic commands that can be given an argument:
\def\nom#1{\leavevmode\hbox{\boldmath$\mathcal{#1}^{\rm N}_{\odot}$}}
\def\Eenom#1{\leavevmode\hbox{\boldmath$\mathcal{#1}^{\rm N}_{Ee}$}}
\def\Epnom#1{\leavevmode\hbox{\boldmath$\mathcal{#1}^{\rm N}_{Ep}$}}
\def\Jenom#1{\leavevmode\hbox{\boldmath$\mathcal{#1}^{\rm N}_{Je}$}}
\def\Jpnom#1{\leavevmode\hbox{\boldmath$\mathcal{#1}^{\rm N}_{Jp}$}}

%%%%%%%%%%%%%%%%
%% Ability to have tables, equations, figures in appendix start from 1, and use appendix section letter.

\newif\ifappletter
\def\apptablenumbers{\global\applettertrue
\setcounter{table}{0}
\setcounter{figure}{0}
\setcounter{equation}{0}
\def\thetable{\thesection\the\c@table}%
\def\fnum@table{{\bf\tablename~\thetable}}%
\def\thefigure{\thesection\the\c@figure}%
\def\fnum@figure{{\bf\figurename~\thefigure}}%
}%

%%% easier to remember than \apptablenumbers
\let\restartappendixnumbering\apptablenumbers

\def\resetapptablenumbers{\global\c@table=0
\global\c@figure=0
\global\c@equation=0
\def\thetable{\thesection\the\c@table}
\def\fnum@table{{\bf\tablename~\thetable}}%
\def\thefigure{\thesection\the\c@figure}
\def\fnum@figure{{\bf\figurename~\thefigure}}%
}

%% written for aastex63
\newif\ifallauthors
\def\allauthors{
\global\allauthorstrue
\let\doauthor\doAllauthors
\ifanonymous
\vskip6pt\vskip1sp\centerline{\large\bf All Authors and
Affiliations\vrule depth 18pt width0pt}\nobreak
\centerline{Anonymous author(s)}
\else
\ifnumlines\nolinenumbers\fi
\onecolumngrid
\clearpage
{\vskip6pt\vskip1sp\centerline{\large\bf All Authors and
Affiliations\vrule depth 18pt width0pt}\nobreak
\global\docollabnum=0
\global\totalentries=0
\global\countauthors=0
\maketitle
}\fi}

%% written for AASTeX62
\def\oldallauthors{%% this conditional keeps \allauthors from turning on
%%                 unless \AuthorCollaborationLimit is used:
\ifnumlines\nolinenumbers\fi
\onecolumngrid
\clearpage
\AuthorCollaborationLimit=10000
%\largestAffilNum=10000 <<== not needed
{\vskip6pt\vskip1sp\centerline{\large\bf All Authors and
Affiliations\vrule depth 18pt width0pt}\nobreak
\maketitle
}}
\let\AllAuthors\allauthors


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Highlight Figure
\definecolor{ltblue}{cmyk}{.5,.1,.1,0}
\newdimen\currwidth
\long\def\highlightfigure#1{
\bgroup
\fboxrule=4pt
\fboxsep=12pt
\vskip6pt
\global\currwidth=\hsize
\global\advance\currwidth by -32pt
\noindent\hbox to\currwidth{\fcolorbox{ltblue}{white}{%
$\vcenter{\hbox to\currwidth{\hss#1\hss}}$}}
\egroup}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% New interactive command:

\def\xlc{lc}
\def\xjs{js}
\def\xanimation{animation}
\newbox\interactbox
\newif\ifcorrectentries


\def\interactive#1#2{
\global\correctentriesfalse\def\checkone{#1}
\ifx\checkone\xlc\correctentriestrue\fi
\ifx\checkone\xjs\correctentriestrue\fi
\ifx\checkone\xanimation \correctentriestrue\fi
\def\checktwo{#2}
\ifx\checktwo\empty\global\correctentriesfalse\fi
\ifcorrectentries
\else\hrule height1pt\vskip12pt\bf ERROR:
The command
{\tt\string\begin\string{interactive\string}\string{\string}\string{\string}} 
needs two arguments. The
first describes the interactive type and the only allowed choices are
lc, js, or animation.\\
The second argument is the movie file or package
containing the interactive figure components. See\\
\url{https://journals.aas.org/aastexguide/\hashmark interactivefig} for more
details.
\vskip12pt\hrule height1pt
\fi \global\currwidth\hsize
\global\advance\currwidth by -32pt
%% we need to set \currwidth here because interactbox expands 
%% before highlightfigure
\global\setbox\interactbox=\vbox\bgroup\hsize=\currwidth\relax\centering
}

\def\endinteractive{\egroup
\ifcorrectentries\highlightfigure{\copy\interactbox}\fi
}

%%%% For cases in which footnotes are called in full width environment
%%   but are used in two cols. This keeps them from overwriting the
%%   second column:

\let\savefootnote\footnote
\def\onecolfootnote#1{\savefootnote{\hsize=.5\textwidth\advance\hsize
by-18pt\relax#1}}

\def\onecolumnfootnotes{\let\footnote\onecolfootnote}

\newdimen\movetableright

%% for anonymous option
\newif\ifsuppressAffiliations
\def\suppressAffiliations{\global\suppressAffiliationstrue}

%% new definition to be used in trackchanges:
\def\sout#1{\raise5pt\hbox{\underline{\vbox to
0pt{\vskip-4pt\hbox{#1\vrule height 8.5pt depth 3.5pt width0pt}\vss}}}} 

%% changes for affiliations when anonymous option is used:
\ifanonymous
\suppressAffiliationstrue
\gdef\affiliation#1{}%% may 20, 2020
\def\xauthor[#1]#2{}
\let\yauthor\@gobble
\def\lookforbracket{\ifx\next[
\let\go\xauthor\else\let\go\yauthor\fi\go}
\def\author{\futurelet\next\lookforbracket}
\fi


%%%%%%%%%%%%%
%% Line numbering


\ifnumlines
\usepackage[mathlines]{lineno}%% dec 20
%%
\linenumbers*[1]
%%\linenumbersep default=10pt
\if@two@col
\advance\linenumbersep -6pt
\else
\advance\linenumbersep 12pt
\fi
\fi

%% Use when line numbers are on, and after \usepackage{amsmath}.
%% Will make line numbering work on math as it would without amsmath.
\def\fixmathlinenumbering{
\let\saveopensqbracket\[
\def\[{\linenomath\saveopensqbracket}
\let\saveequation\equation
\let\saveendequation\endequation
%%
 \let\savealign\align
 \let\saveendalign\endalign
%%
 \let\savealigned\aligned
 \let\saveendaligned\endaligned
%%
\def\align{\linenomath\savealign}
 \def\endalign{\saveendalign}
 \def\aligned{\linenomath\savealigned}
 \def\endaligned{\linenomath\saveendaligned}
%%
\def\equation{\linenomath\saveequation}
\def\endequation{\linenomath\saveendequation}
\usepackage[mathlines]{lineno}
}

\AtBeginDocument{\expandafter\ifx\csname
@amsmath@err\endcsname\relax\else\fixmathlinenumbering\fi}

\endinput
%% Change Log
============================
Jan 29, 2021
added \ifonecolstyle to see if reference numberlines need to
be closer to text

Dec 20, 2020
Line numbering solutions:

Line numbering equations and \[ \] works fine.

$$math $$ does not work, and should be replaced with \[ math \]

If author has \usepackage{amsmath} the equations will not number;
amsmath and lineno are incompatible.


June 28
For using \edit{}{} in section heads; new command,
\simpleedit that will prevent errors in section heads
and in bookmarks. \edit is set to be equal to \simpleedit{}{}
in \section definition, so nothing needs to be done by user.
\protect used in section head, as \protect\edit{}{}, is not
necessary.

June 27
\null added to \end{deluxetable} and \csname enddeluxetable*\endcsname
to force final page of multipage table to print.

June 26
Acknowledgments: remember to use
\begin{acknowledgments}...\end{acknowledgments} form
OR
\begin{acknowledgements}...\end{acknowledgements}

Now have an error message built in if someone uses
just \acknowledgments or \acknowledgements


June 4, 2020
   Added \nocollaboration{0} to first
   section head, if anonymous was true. This
   allowed correct titlepage for anonymous
   if any style option was used.

Acknowledgments: remember to use
\begin{acknowledgments}...\end{acknowledgments} form




%% Change Log
============================
May 19, 2020
Changed definition of \sout (strike out) so that
it works with a variety of citations in
argument of \deleted and \replaced without complaint.

Made anonymous work with \longauthor, and without
\collaboration or \nocollaboration.

Changed \acknowledgements to
\begin{acknowledgements}...\end{acknowledgements}
(More error proof, and standard mark up)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++

June 8, 2019
Changed \author and \collaboration macros.
Collaboration now takes two arguments:
#1= number of authors to be listed before the
name of the collaboration; #2 = name of the collaboration

\nocollaboration{} has one argument, 
#1= the number of authors above it that we want to print on the title page.

June 6, 2019

\acknowledgments does not use \begin{}...\end{} form,
now just \acknowledgments command. Acknowledgments text
ends with \par, so if author wants more than one paragraph
in acknowledgment he/she should surround text with curly
bracket:

\acknowledgments
text...

or

\acknowledgments
{text...

more text...}

------


============================
June 3, 2019
\let\footnote\onecolfootnote on page where references start, but
before footnotes are entered,
will allow footnotes to break in the right place (one column's width).

============================
May 15, 2019
\centerwidetable is now \movetableover at Greg's suggestion.

============================
Nov 27, 2017

1) Change to \begin{widetext}...\end{widetext} to
return the horizontal lines before and after the 
environment.

2) \NoNewPageAfterKeywords is now default;
authors wanting a new page after keywords can
use the \NewPageAfterKeywords command.

3) Change in definition names, returning to earlier version:
\turnoffediting will turn off colors for all levels
of editing mark up made with \edit1{}, \edit2{}, or \edit3{}.
You can turn off any level of editing with 
\turnoffeditone, \turnoffedittwo, and \turnoffeditthree.

4) It is now possible to use \doi{} as an argument of \dataset.
\dataset[]{\doi{10.5281/zenodo.831784}} and
\dataset[\doi{10.5281/zenodo.831784}]{\doi{10.5281/zenodo.831784}}
will work.
 
============================
October 15, 2017

1) \movetabledown now works with longrotatetable as it does with
rotatetable.
\movetabledown=<dimen> before either 
\begin{rotatetable} or
\begin{longrotatetable}

Added \clearpage before \begin{rotatetable} and \begin{rotatetable*}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
2)

\NoNewPageAfterKeywords, used after \documentclass{AASTeX62},
will keep a new page from starting after keywords.
Extra vertical space before first section in Modern option.

Solution suggested by  Alex Drlica-Wagner.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
3)\edit1{}, \edit2{} and \edit3{} will add color
to the edited text.

To get rid of the color, but not the text, authors should
use \turnoffeditonecolor (for \edit1{})
 \turnoffedittwocolor (for \edit2{})
 \turnoffeditthreecolor (for \edit3{})
These commands have the same results as
\turnoffeditone, \turnoffedittwo, and \turnoffeditthree
which are not disabled, but the new commands make the
action to be produced more descriptive.

Similarly, instead of \turnoffedit or \turnoffedits
we now have \turnoffeditcolors. The original two
commands still work, but the new command is more
descriptive of the results.

Authors should be instructed to put these commands before
\begin{document}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
4)
\begin{widetext}...\end{widetext}

\begin{widetext}
will go from twocolumngrid to onecolumngrid, centered.

\end{widetext} will turn off centering and go back
to twocolumngrid.

Only needed in twocolumn styles; nothing will happen
in single column styles.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
5) Sean Lake's problem with \mag was solved by Barbara Beeton online,

``the latest edit identifies the problem unambiguously: \mag is a tex
primitive, and shouldn't be redefined. why this didn't appear in the
logs you posted is beyond me, but if you rename your command to (say)
\Mag it should stop having a problem. � barbara beeton Feb 22''

\mag is a tex primitive, and it is expecting to be followed by a
number.

This had nothing to do with AASTeX61.cls

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
6) 
\def\arraystretch{.9} as requested will make the space between
lines in tables shrink to 90% of the original space.

I think what is really wanted is \def\arraystretch{1.1} which will
still give us 110% of the original space, a little extra vertical space, but not
as much as we had before.

Meanwhile, an author can change the definition of \arraystretch{}
for a single table or all the tables in his paper, if he wants more
or less space.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
7) Abstract is now the same width as in emulateapj
in all styles, with the exception of the modern option 
which is unchanged. Abstracts in all styles will break over pages
in this version.

Thanks to suggestion by Patricio Cubillos.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
8) Center and tighten affiliations, changed and
tested for all styles, including longauthor option.

samples.zip show examples titlepage of each style with and
without longauthor option.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
9) preprintnumbers activated.
No need to use a documentclass option.

Just enter \reportnum{<text>}, before \begin{document}
and repeat as many times as needed. The entries will stack 
in the top right hand corner of the titlepage, for all styles.

ie
\reportnum{YOUR-INST-REP-789}
\reportnum{MY-INST-REP-123}

%% author can continue with as many entries as desired.
% \reportnum{A number of other numbers}
% \reportnum{More Numbers}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
10) Title is no longer uppercase by default. It is
now title case, which means all important words are
capitalized by author.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
11) \AuthorCollaborationLimit now
set to default of 10,000. If smaller number of
authors wanted, \AuthorCollaborationLimit can
be set to the number wanted; \allauthors typed
at end of article will make every author and
affiliation print.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
12) \startlongtable\deluxetable now works in appendix, even
when it is the last entry in the paper. Added \null
to \endlongdeluxetable which has the effect of making
last page print.

This works in my test, but it
may be useful to see the example from author, github #54

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
13) Deluxetable no longer crashes if author has neglected
to put in \\ before \enddata. I added the non-redundant \\
command, \crcr, and sample now works.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
14) \tablehead in deluxetable had too much space underneath
double horizontal lines. Made custom negative skip based 
on each style, tested, and supplied tabsamples.zip showing
tableheads in each style.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
15) \urlstyle{rm} is set so that url and href produce
the same fonts.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
16) Fix to \dataset so that square brackets
are not printed in the case of argument
in square brackets.

Now
\dataset[]{http://www.texnology.com}
produces
[DATASET] linked to www.texnology.com

\dataset[TeXnology]{http://www.texnology.com}
produces
TeXnology linked to www.texnology.com

%% This will also work,
% same results as \dataset[]{http://www.texnology.com}
% but using the more common LaTeX syntax:

\dataset{http://www.texnology.com}
also produces
[DATASET] linked to www.texnology.com

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
17) New version of DOI:

\doi{10.5281/zenodo.831784}
now produces hyperlinked

https://doi.org/10.5281/zenodo.831784

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
18) Added to definition of \software,
working with sample from August Muench, reduced
the space between words, and got rid of
extra space after punctuation with the
\frenchspacing command.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
19) August Muench entered a bug report
complaining that \clearpage doesn't always
work, but he didn't have a sample, so I am
not able to debug this one. It may be that
the problem isn't with the code, but with
the way the commands were entered, but 
without a sample I have no way of knowing.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
20)
Tested \startlongtable in all styles
and didn't find a problem with centering.
I did add a skip at the beginning of
\startlongtable in case someone didn't
leave a blank line before using the
command.

============================
September 21, 2016
Finishing Version 6.1 update

=====================
August 27, 2016                               
Many changes

=====================
August 17
Many changes:

%% note:
%% this keeps tables from being set to \small, code used in Revtex4-1
\let\table@hook\relax

=====================
Feb 29, changes to table macros to center caption even if table is wider than
width of text; give error message if user asks to rotate table but
forgets to use \floattable before table; make rotating table be full
text width even if called for in double columns.

March 1, reworking figure and table numbering for appendices. 
Deleted recent definition of \appendix, returned to earlier version
with additions.

================






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% New interactive command:

\def\xlc{timeseries}
\def\xjs{js}
\def\xanimation{animation}
\newbox\interactbox
\newif\ifcorrectentries

\def\interactive#1#2{
\global\correctentriesfalse\def\checkone{#1}
\ifx\checkone\xlc\correctentriestrue\fi
\ifx\checkone\xjs\correctentriestrue\fi
\ifx\checkone\xanimation \correctentriestrue\fi
\ifcorrectentries
\else\hrule height1pt\vskip12pt\bf ERROR: Your choices for the first argument for
{\tt\string\interactive} are timeseries, js, or animation.\\[6pt] Please choose one
of these terms.\vskip12pt \fi
%%%
\def\checktwo{#2}
\ifx\checktwo\empty \vskip12pt \bf ERROR: The command {\tt\string\interactive} needs
two arguments, with the second argument for the graphics file or files
needed.\vskip12pt\hrule height1pt \else\global\correctentriestrue\fi
\global\currwidth\hsize
\global\advance\currwidth by -32pt
%% we need to set \currwidth here because interactbox expands 
%% before highlightfigure
\global\setbox\interactbox=\vbox\bgroup\hsize=\currwidth\centering}

\def\xxinteractive#1#2{\global\correctentriesfalse\def\checkone{#1}
\ifx\checkone\xlc\correctentriestrue\fi
\ifx\checkone\xjs\correctentriestrue\fi
\ifx\checkone\xanimation \correctentriestrue\fi
\ifcorrectentries
\else\hrule height1pt\vskip12pt\bf 
\noindent ERROR:\\[6pt] Your choices for the first argument for
{\tt\string\begin\string{\string\interactive\string}} are timeseries, js, or animation.\\[6pt] Please choose one
of these terms.\vskip12pt \fi
%%%
\def\checktwo{#2}
\ifx\checktwo\empty \vskip12pt \bf 
\noindent ERROR:\\[6pt]
The second argument of %{\tt\string\begin\string{interactive\string}} is the file containing the
interactive component. Please check your LaTeX to be sure it conforms.
See %\href{https://journals.aas.org/aastexguide/interactivefig}{stuff}
%{https://journals.aas.org/aastexguide/\#interactivefig}
for more details.
\vskip12pt\hrule height1pt \else\global\correctentriestrue\fi
\global\currwidth\hsize
\global\advance\currwidth by -32pt
%% we need to set \currwidth here because interactbox expands 
%% before highlightfigure
\global\setbox\interactbox=\vbox\bgroup\hsize=\currwidth\centering}

\def\endinteractive{\egroup
\ifcorrectentries\highlightfigure{\copy\interactbox}\fi}

