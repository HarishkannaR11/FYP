% ssnreview.cls -- Undergraduate Project Review Report class
% Version 1.0 (2026/08/07)
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ssnreview}[2026/08/07 v1.0 SSN Project Review Report]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass[12pt,a4paper]{report}

\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage[a4paper,top=1in,bottom=1in,left=1.25in,right=1in]{geometry}
\RequirePackage{setspace}
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{xcolor}
\RequirePackage{tikz}
\RequirePackage{pgfgantt}
\RequirePackage{caption}
\RequirePackage{enumitem}
\RequirePackage{titlesec}
\RequirePackage[hidelinks]{hyperref}

\onehalfspacing
\setlength{\parindent}{0.5in}
\setlength{\parskip}{3pt}
\renewcommand{\bibname}{References}
\titleformat{\chapter}[hang]{\normalfont\bfseries\Large}{\thechapter.}{0.5em}{}
\titlespacing*{\chapter}{0pt}{-20pt}{18pt}

% Metadata defaults and user commands
\newcommand{\RR@title}{TITLE OF THE PROJECT}
\newcommand{\RR@review}{REVIEW REPORT -- I}
\newcommand{\RR@students}{Student Name (Register No.)}
\newcommand{\RR@supervisor}{Dr./Mr./Ms. Supervisor Name}
\newcommand{\RR@department}{Department of Computer Science and Engineering}
\newcommand{\RR@institution}{Sri Sivasubramaniya Nadar College of Engineering}
\newcommand{\RR@academicyear}{2026--2027}
\newcommand{\RR@logo}{ssn-logo.png}

\newcommand{\projecttitle}[1]{\renewcommand{\RR@title}{#1}}
\newcommand{\reviewnumber}[1]{\renewcommand{\RR@review}{#1}}
\newcommand{\students}[1]{\renewcommand{\RR@students}{#1}}
\newcommand{\supervisor}[1]{\renewcommand{\RR@supervisor}{#1}}
\newcommand{\department}[1]{\renewcommand{\RR@department}{#1}}
\newcommand{\institution}[1]{\renewcommand{\RR@institution}{#1}}
\newcommand{\academicyear}[1]{\renewcommand{\RR@academicyear}{#1}}
\newcommand{\collegelogo}[1]{\renewcommand{\RR@logo}{#1}}

\newcommand{\makereviewtitle}{%
  \begin{titlepage}
    \begin{center}
      \IfFileExists{\RR@logo}
        {\includegraphics[width=3.2cm]{\RR@logo}\\[0.4cm]}
        {\fbox{\parbox[c][2.4cm][c]{3.2cm}{\centering SSN\\College Logo}}\\[0.4cm]}
      {\Large\bfseries \RR@institution\par}\vspace{0.2cm}
      {\large \RR@department\par}
      \vfill
      {\Large\bfseries Title\par}\vspace{0.5cm}
      {\LARGE\bfseries \RR@title\par}\vspace{0.8cm}
      {\Large\bfseries \RR@review\par}
      \vfill
      {\large\bfseries Submitted by\par}\vspace{0.3cm}
      {\large\begin{tabular}{c}\RR@students\end{tabular}\par}
      \vspace{0.8cm}
      {\large\bfseries Under the guidance of\par}\vspace{0.2cm}
      {\large \RR@supervisor\par}
      \vfill
      {\large Academic Year: \RR@academicyear\par}
    \end{center}
  \end{titlepage}
}

\newcommand{\reviewfrontmatter}{%
  \pagenumbering{roman}
  \tableofcontents
  \listoffigures
  \listoftables
  \clearpage
  \pagenumbering{arabic}
}

\newenvironment{reviewabstract}
  {\chapter*{Abstract}\addcontentsline{toc}{chapter}{Abstract}}
  {}

\newcommand{\keywords}[1]{\par\noindent\textbf{Keywords:} #1}

\endinput