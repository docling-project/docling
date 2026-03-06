### ibm.com /redbooks

### of duties the database column masks

### Leverage row permissions on Protect columns by defining

### Implement roles and separation

## Front cover

## Support in IBM DB2 for i

## Red paper

Jim Bainbridge Hernando Bedoya Rob Bestgen Mike Cain Dan Cruikshank Jim Denton Doug Mack Tom McKinley Kent Milligan

## Row and Column Access Control

<!-- image -->

© Copyright IBM Corp. 2014. All rights reserved.

Notices . . vii Trademarks . . . viii Preface . . xi . . . . . . xi xiii x iii xiv 1 2 2 3 4 5 7 2.1 Roles . . . . . 8 8 8 9 9 9 10 . . . . . 10 10 13 14 14 16 18 18 19 20 21 22 22 23 23 24 25 26 28 29 32

### 2.2 Separation of duties 3.2.1 Special registers 3.6.6 Activating RCAC

## Contents

DB2 for i Center of Excellence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1.3 DB2 for i security controls 1.3.1 Existing row and column control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.2.2 Built-in global variables 3.4 Establishing and controlling a . . . . . . . . . . . . . . . . . . . . . . . . 3.6 Human resources example . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.6.5 Defining and creating column masks

Now you can become a published author, too! . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Chapter 2. Roles and separation of duties . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Chapter 3. Row and Column Access Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.2 Special registers and built-in global variables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.6.3 Demonstrating data access without RCAC 3.6.4 Defining and creating row permissions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

Chapter 1. Securing and protecting IBM DB2 data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.1.1 Row permission and column mask definitions

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2.1.4 Database Information function: QIBM_DB_SYSMON . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ccessibility by using the RCAC rule text. . . . . . . . . . . . . 3.5 SELECT, INSERT, and UPDATE behavior with RCAC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ix

. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Authors. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Comments welcome. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Stay connected to IBM Redbooks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1.1 Security fundamentals. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1.2 Current state of IBM i security. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1.3.2 New controls: Row and Column Access Control. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2.1.1 DDM and DRDA application server access: QIBM_DB_DDMDRDA . . . . . . . . . . . 2.1.2 Toolbox application server access: QIBM_DB_ZDA. . . . . . . . . . . . . . . . . . . . . . . . 2.1.3 Database Administrator function: QIBM_DB_SQLADM . . . . . . . . . . . . . . . . . . . . . 2.1.5 Security Administrator function: QIBM_DB_SECADM . . . . . . . . . . . . . . . . . . . . . . 2.1.6 Change Function Usage CL command. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2.1.7 Verifying function usage IDs for RCAC with the FUNCTION_USAGE view 3.1 Explanation of RCAC and the concept of access control . . . . . . . . . . . . . . . . . . . . . . . 3.1.2 Enabling and activating RCAC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.3 VERIFY_GROUP_FOR_USER function. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.6.1 Assigning the QIBM_DB_SECADM function ID to the consultants. . . . . . . . . . . . 3.6.2 Creating group profiles for the users and their roles. . . . . . . . . . . . . . . . . . . . . . . 3.6.7 Demonstrating data access with RCAC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.6.8 Demonstrating data access with a view and RCAC . . . . . . . . . . . . . . . . . . . . . . .

iii

<!-- image -->

"	#

""

Highlights

DB2 for i Center of Excellence

IBM Systems Lab Services and Training Solution Brief

Global CoE engagements cover topics including:

r

Database performance and scalability

r

Advanced SQL knowledge and skills transfer

r

Business intelligence and analytics

r

DB2 Web Query

r r

Database modernization and re-engineering

r

Data-centric architecture and design

r r

ISV education and enablement

Extremely large database and overcoming limits to growth

Query/400 modernization for better reporting and analysis capabilities

!

!	#

Who we are, some of what we do

We build confident, satisfied clients No one else has the vast consulting experiences, skills sharing and renown service offerings to do what we can do for you. Because no one else is IBM. With combined experiences and direct access to development groups, we're the experts in IBM DB2® for i. The DB2 for i Center of Excellence (CoE) can help you achieve—perhaps reexamine and exceed—your business requirements and gain more confidence and satisfaction in IBM product data management products and solutions.

Expert help to achieve your business requirements

## DB2 for i Center of Excellence

Power Services

<!-- image -->

© Copyright IBM Corp. 2014. All rights reserved.

Jim Bainbridge is a senior DB2 consultant on the DB2 for i Center of Excellence team in the IBM Lab Services and Training organization. His primary role is training and implementation services for IBM DB2 Web Query for i and business analytics. Jim began his career with IBM 30 years ago in the IBM Rochester Development Lab, where he developed cooperative processing products that paired IBM PCs with IBM S/36 and AS/.400 systems. In the years since, Jim has held numerous technical roles, including independent software vendors technical support on a broad range of IBM technologies and products, and supporting customers in the IBM Executive Briefing Center and IBM Project Office. Hernando Bedoya is a Senior IT Specialist at STG Lab Services and Training in Rochester, Minnesota. He writes extensively and teaches IBM classes worldwide in all areas of DB2 for i. Before jo ining STG Lab Services, he worked in the ITSO for nine years writing multiple IBM Redbooks® publications. He also worked for IBM Colombia as an IBM AS/400® IT Specialist doing presales support for the Andean countries. He has 28 years of experience in the computing field and has taught database classes in Colombian universities. He holds a Master's degree in Computer Science from EAFIT, Colombia. His areas of expertise are database technology, performance, and data warehousing. Hernando can be contacted at hbedoya@us.ibm.com .

### Authors

## Preface

function and advantages of co ntrolling access to data in a e capabilities of RCAC an d provides examples database environment.

comprehensive and transparent governance policy. A solid background in IB database concepts, and SQL is assumed.

way. This publication helps you understand th

xi

This IBM® Redpaper™ publication provides information about the IBM i 7.2 feature of IBM M i object level security, DB2 for i relational

This paper was produced by the IBM DB2 for i Center of Excellence team in partnership with the International Technical Support Organization (ITSO), Rochester, Minnesota US.

DB2® for i Row and Column Access Control (RCAC). It offers a broad description of the of defining, creating, and implementing the row permissions and column masks in a relational This paper is intended for database engineers, data-centric application developers, and security officers who want to design and implement RCAC as a part of their data control and

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

© Copyright IBM Corp. 2014. All rights reserved.

1

http://www.idtheftcenter.org 2

http://www.ponemon.org /

Recent news headlines are fille d with reports of data breache s and cyber-attacks impacting

### Chapter 1.

## data

1

global businesses of all sizes. The Identity Theft Resource Center 1 reports that almost 5000 data breaches have occurred since 2005, expo sing over 600 million records of data. The financial cost of these data breaches is skyr ocketing. Studies from the Ponemon Institute

2 resulted in a brand equity loss of $9.4 million per attack. The aver age cost that is incurred for Businesses must make a seriou s effort to secure their data and recognize that securing longer an option; it is a requirement. are covered in this chapter: Security fundamentals Current state of IBM i security DB2 for i security controls

## Securing and protecting IBM DB2

## 1

revealed that the average cost of a data breach increased in 2013 by 15% globally and each lost record containing sensitive information increased more than 9% to $145 per record. information assets is a cost of doing business. In many parts of the world and in many industries, securing the data is required by law and subject to audits. Data security is no This chapter describes how you can secure and protect data in DB2 for i. The following topics

<!-- image -->

## 2 Row and Column Access Control Support in IBM DB2 for i

Because of the inherently secure nature of IBM i, many clients rely on the default system settings to protect their business data that is stored in DB2 for i. In most cases, this means no data protection because the default setting for the Create default public authority (QCRTAUT) system value is *CHANGE. Even more disturbing is that many IBM i clients remain in this state, despite the news headlines and the significant costs that are involved with databases being compromised. This default security configuration makes it quite challenging to implement basic security policies. A tighter implementation is required if you really want to protect one of your company's most valuable assets, which is the data. Traditionally, IBM i applications have employed menu-based security to counteract this default configuration that gives all users access to the data. The theory is that data is protected by the menu options controlling what database op erations that the user can perform. This approach is ineffective, even if the user profile is restricted from running interactive commands. The reason is that in today's connected world there are a multitude of interfaces into the system, from web browsers to PC clients, that bypass application menus. If there are no object-level controls, users of these newer interfaces have an open door to your data.

### 1.2 Current state of IBM i security

resource security. If implemented properly, resource security prevents data breaches from both internal and external intrusions. Resource security controls are closely tied to the part of the security policy that defines who should have access to what information resources. A hacker might be good enough to get through your company firewalls and sift his way through to your system, but if they do not have explicit access to your database, the hacker cannot compromise your information assets. With your eyes now open to the importance of securing information assets, the rest of this chapter reviews the methods that are available for securing database resources on IBM i.

security policy. Without a security policy, there is no definition of what are acceptable practices for using, accessing, and storing information by who, what, when, where, and how. A security policy should minimally address three things: confidentiality, integrity, and availability. The monitoring and assessment of adherence to the security policy determines whether your security strategy is working. Often, IBM security consultants are asked to perform security assessments for companies without regard to the security policy. Although these assessments can be useful for observing how the system is defined and how data is being accessed, they cannot determine the level of security without a security policy. Without a security policy, it really is no t an assessment as much as it is a baseline for monitoring the changes in the security settings that are captured. A security policy is what defines whether the system and its settings are secure (or not). The second fundamental in securing data assets is the use of

Before reviewing database security techniques, there are two fundamental steps in securing information assets that must be described: First, and most important, is the definition of a company's

### 1.1 Security fundamentals

<!-- image -->

<!-- image -->

4

_Figure 1-2 Existing row and column controls_

Row and Column Access Control Support in IBM DB2 for i

User with *ALLOBJ access

gic, as shown in Figure 1-2. However, such as Open Database Connectivity (ODBC) and System i Navigator. ormance and management issues, a user with

views (or logical files) and application lo are provided by the IBM i operating system, Even if you are willing to live with these perf *ALLOBJ access still can directly access all of th e data in the underlying DB2 table and easily

#### 1.3.1 Existing row and column control

perform their job. Often, users with object-lev el access are given access to row and column values that are beyond what their business ta sk requires because that object-level security only for the employees that they manage.

as the amount of data grows and the number of users increases. bypass the security controls that are built into an SQL view.

Some IBM i clients have tried augmenting the all-or-nothing object-level security with SQL application-based logic is easy to bypass with all of the different data access interfaces that Using SQL views to limit access to a subset of the data in a table also has its own set of challenges. First, there is the complexity of managing all of the SQL view objects that are used for securing data access. Second, scaling a view-based security solution can be difficult

Many businesses are trying to limit data access to a need-to-know basis. This security goal means that users should be given access only to the minimum set of data that is required to provides an all-or-nothing solution. For example, object-level controls allow a manager to access data about all employees. Most security policies limit a manager to accessing data

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

### 2.2 Separation of duties

ndividuals without overl apping responsibilities,

SELECT function_id, user_name, usage, user_type ORDER BY user_name;

shown in Example 2-1.

FROM function_usage

Column name Data type Description FUNCTION_ID VARCHAR(30) USER_NAME VARCHAR(10) function. USAGE VARCHAR(7) Usage setting: USER_TYPE VARCHAR(5)

ID of the function. Name of the user pr Type of user profile:

_Table 2-1 FUNCTION_USAGE view_

ofile that has a usage setting for this USER: The user profile is a user. GROUP: The user profile is a group.

#### 2.1.7 Verifying function usage IDs

2-1 describes the columns in the FUNCTION_USAGE view.

WRKFCNUSG ) CHGFCNUSG ) DSPFCNUSG ) CHGFCNUSG

Work Function Usage ( Change Function Usage ( Display Function Usage ( For example, the following

used to prevent fraudulent activities or errors by a single person. It provides the ability for administrative functions to be divided across i

## 10 Row and Column Access Control Support in IBM DB2 for i

WHERE function_id='QIBM_DB_SECADM'

#### 2.1.6 Change Function Usage CL command

HBEDOYA to administer and manage RCAC rules:

#### for RCAC with the FUNCTION_USAGE view

command shows granting authorization to user CHGFCNUSG FCNID(QIBM_DB_SECADM) USER(HBEDOYA) USAGE(*ALLOWED)

ALLOWED: The user profile is allowed to use the function. DENIED: The user profile is not allowed to use the function.

Separation of duties helps businesses comply with industry regulations or organizational requirements and simplifies the management of authorities. Separation of duties is commonly so that one user does not possess unlimited authority, such as with the *ALLOBJ authority.

To discover who has authorization to define and manage RCAC, you can use the query that is

Example 2-1 Query to determine who has authority to define and manage RCAC

The following CL commands can be used to work with, display, or change function usage IDs:

The FUNCTION_USAGE view contains function usage configuration details. Table

<!-- image -->

SET CURRENT DEGREE (SQL statement) XX CHGQRYA XX STRDBMON or ENDDBMON XX STRDBMON or ENDDBMON i Navigator's SQL Details for Job Visual Explain within Run SQL scripts XX ANALYZE PLAN CACHE procedure XX DUMP PLAN CACHE procedure XX MODIFY PLAN CACHE procedure XX XX XX

User action

command targeting a different user's job QUSRJOBI() API format 900 or System Visual Explain outside of Run SQL scripts

commands targeting a different user's job XXX

commands targeting a job that matches the current user XXXX XXXX

### Chapter 2. Roles and separation of duties 11

Theresa. Before release IBM i 7.2, to grant privileges, Theresa had to have the same privileges Theresa was granting to others. Therefore, to grant *USE privileges to the PAYROLL table, Theresa had to have *OBJMGT and *USE authority (or a higher level of authority, such as *ALLOBJ). This requirement allowed Theresa to access the data in the PAYROLL table even though Theresa's job description was only to manage its security. table. QIBM_DB_SECADM function usage can be granted only by a user with *SECADM special authority and can be given to a user or a group. QIBM_DB_SECADM also is responsible for admi nistering RCAC, which restricts which rows a user is allowed to access in a table and whether a user is allowed to see information in certain columns of a table. A preferred practice is that the RCAC administrator has the QIBM_DB_SECADM function usage ID, but absolutely no other data privileges. The result is that the RCAC administrator can deploy and maintain the RCAC constructs, but cannot grant themselves unauthorized access to data itself. Table 2-2 shows a comparison of the different function usage IDs and *JOBCTL authority to the different CL commands and DB2 for i tools.

For example, assume that a business has assigned the duty to manage security on IBM i to In IBM i 7.2, the QIBM_DB_SECADM function usage grants authorities, revokes authorities, changes ownership, or changes the primary group without giving access to the object or, in the case of a database table, to the data that is in the table or allowing other operations on the

_Table 2-2 Comparison of the different function usage IDs and *JOBCTL authority_

MODIFY PLAN CACHE PROPERTIES procedure (currently does not check authority) CHANGE PLAN CACHE SIZE procedure (currently does not check authority)

*JOBCTL QIBM_DB_SECADM QIBM_DB_SQLADM QIBM_DB_SYSMON No Authority

<!-- image -->

<!-- image -->

ENABLE DISABLE ;

FOR ALL ACCESS

WHERE < >

AS < >

ON < >

< >

_table name_

FOR ROWS

ENFORCED

Column mask

correlation name

identification number.

CREATE PERMISSION permission name

_Figure 3-1 CREATE PE RMISSION SQL statement_

The SQL

### Chapter 3. Row and Column Access Control 15

A column mask is a database object that manifests a column value access control rule for a specific column in a specific table. It uses a CASE expression that describes what you see when you access the column. For example, a teller can see only the last four digits of a tax

## S ifi th pec t th i i i t b ifi i iti ll es that the row permission is to be initially di bl d sabled

logic to test: user and/or group and/or column value

Specifies that the row permission applies to all references of the table FOR ALL ACCESS

Specifies that the row permission is to be initially enabled

CREATE PERMISSION

Identifies the table on which the row permission is created

Specifies an optional correlation name that can be used within search-condition

Indicates that a row permission is created Specifies a condition that can be true, false, or unknown

Names the row permission for row access control

statement that is shown in Figure 3-1 is used to define and initially enable or disable the row access rules.

#### 3.2.2 Built-in global variables

USER = ALICE CURRENT USER = JOE

USER = ALICE CURRENT USER = ALICE

CALL proc1

## P1 Proc1: Owner = JOE SET OPTION USRPRF=*OWNER

Signed on as ALICE

USER = ALICE CURRENT USER = ALICE

_Figure used:_

Special register Co USER or SESSION_USER CURRENT_USER SYSTEM_USER that initiated

rresponding value The authorization ID the connection.

SignedonasALICE

The effective user of the th The effective user of the thread

While the procedure is running, the special register USER still

read excluding adopted authority.

contains the value of ALICE USER having the value of ALICE.

_Table 3-1 Special registers and their corresponding values_

including adopted authority. When no adopted authority is present, this has the same value as USER.

_Figure 3-5 Special registers and adopted authority_

database connection and used as part of the RCAC logic.

A user connects to the server using the user profile ALICE. USER and CURRENT USER initially have the same value of ALICE. and was created to adopt JOE's authority when it is called. contains the value of JOE because it includes any adopted authority.

statements to retrieve scalar values that are associated with the variables.

### Chapter 3. Row and Column Access Control

Built-in global variables are provided with the database manager and are used in SQL IBM DB2 for i supports nine different built-in global variables that are read only and maintained by the system. These global variables can be used to identify attributes of the

Table

3-5 shows the difference in the special register values when an adopted authority is ALICE calls an SQL procedure that is named proc1, which is owned by user profile JOE because it excludes any adopted authority. The special register CURRENT USER When proc1 ends, the session reverts to its original state with both USER and CURRENT

19

3-1 summarizes these special registers and their values.

<!-- image -->

1. 2. 3.

Global variable Type Description CLIENT_HOST VARCHAR(255) Host CLIENT_IPADDR VARCHAR(128) CLIENT_PORT INTEGER PACKAGE_NAME VARCHAR(128) VARCHAR(128) VARCHAR(64) Version identi ROUTINE_SCHEMA VARCHAR(128) VARCHAR(128) Name ROUTINE_TYPE CHAR(1)

Table

IP address of the PACKAGE_SCHEMA PACKAGE_VERSION

_Table 3-2 Built-in global variables name of the current client ROUTINE_SPECIFIC_NAME_

as returned by the system fier of the currently running package of the currently running routine Type of the currently running routine

Here is an example of using th invocations return a value of 1:

current client as returned by the system Name of the currently running package Schema name of the currently running package Schema name of the currently running routine

3-2 lists the nine built-in global variables.

The first parameter must be one of these th of 0. It never returns the null value.

## 20 Row and Column Access Control Support in IBM DB2 for i

ree special registers: SESSION_USER, USER, or do not exist without receiving any kind of error. e VERIFY_GROUP_FOR_USER function: VERIFY_GROUP_FOR_USER (CURRENT_USER, 'MGR') The following function invocation returns a value of 0:

Port used by the current client to communicate with the server

There are user profiles for MGR, JANE, JUDY, and TONY. The user profile JANE specifies a group profile of MGR. VERIFY_GROUP_FOR_USER (CURRENT_USER, 'JANE', 'MGR') VERIFY_GROUP_FOR_USER (CURRENT_USER, 'JUDY', 'TONY')

VERIFY_GROUP_FOR_USER (CURRENT_USER, 'JANE', 'MGR', 'STEVE')

### 3.3 VERIFY_GROUP_FOR_USER function

The VERIFY_GROUP_FOR_USER function was added in IBM i 7.2. Although it is primarily intended for use with RCAC permissions and masks, it can be used in other SQL statements. CURRENT_USER. The second and subsequent parameters are a list of user or group profiles. Each of these values must be 1 - 10 characters in length. These values are not validated for their existence, which means that you can specify the names of user profiles that If a special register value is in the list of user profiles or it is a member of a group profile included in the list, the function returns a long integer value of 1. Otherwise, it returns a value If a user is connected to the server using user profile JANE, all of the following function

<!-- image -->

<!-- image -->

RETURN CASE END ENABLE ;

FOR COLUMN TAX_ID ELSE 'XXX-XX-XXXX'

THEN EMPLOYEES . TAX_ID THEN EMPLOYEES . TAX_ID THEN EMPLOYEES . TAX_ID

Example 3-9 Creating a mask on the TAX_ID column

AND SESSION_USER = EMPLOYEES . USER_ID

CREATE MASK HR_SCHEMA.MASK_TAX_ID_ON_EMPLOYEES ON HR_SCHEMA.EMPLOYEES AS EMPLOYEES AND SESSION_USER <> EMPLOYEES . USER_ID

### Chapter 3. Row and Column Access Control 27

RETURN CASE END 2. – – – –

ELSE NULL ENABLE ; 3-9.

the X character (for example, XXX-XX-1234).

THEN EMPLOYEES . DATE_OF_BIRTH THEN EMPLOYEES . DATE_OF_BIRTH Any other person sees the entire T

The other column to mask in this example is rules to enforce include the following ones:

AND SESSION_USER = EMPLOYEES . USER_ID AND SESSION_USER <> EMPLOYEES . USER_ID DAY (EMPLOYEES.DATE_OF_BIRTH )) the TAX_ID information. In this example, the Employees can see only their own unmasked TAX_ID. AX_ID as masked, for example, XXX-XX-XXXX.

WHEN VERIFY_GROUP_FOR_USER ( SESSION_USER , 'MGR' ) = 1 WHEN VERIFY_GROUP_FOR_USER ( SESSION_USER , 'MGR' ) = 1

WHEN VERIFY_GROUP_FOR_USER ( SESSION_USER , 'HR' ) = 1 WHEN VERIFY_GROUP_FOR_USER ( SESSION_USER , 'MGR' ) = 1 WHEN VERIFY_GROUP_FOR_USER ( SESSION_USER , 'MGR' ) = 1 WHEN VERIFY_GROUP_FOR_USER ( SESSION_USER , 'EMP' ) = 1

THEN ( 'XXX-XX-' CONCAT QSYS2 . SUBSTR ( EMPLOYEES . TAX_ID , 8 , 4 ) )

WHEN VERIFY_GROUP_FOR_USER ( SESSION_USER , 'HR', 'EMP' ) = 1 THEN ( 9999 || '-' || MONTH ( EMPLOYEES . DATE_OF_BIRTH ) || '-' || Human Resources can see the unmasked TAX_ID of the employees. Managers see a masked version of TAX_ID with the first five characters replaced with To implement this column mask, run the SQL statement that is shown in Example

<!-- image -->

<!-- image -->

_Figure 3-11 Selecting the EMPL OYEES table from System i Navigator_

2. in Figure Schemas   Tables , right-click the EMPLOYEES Definition .

Look at the definitio HR_SCHEMA table, and click

ALTER TABLE HR_SCHEMA.EMPLOYEES ACTIVATE ROW ACCESS CONTROL ACTIVATE COLUMN ACCESS CONTROL; 3-11. To do this, from

#### 3.6.6 Activating RCAC

1. 3-10.
3. Figure

_Figure 3-10 Column masks shown in System i Navigator_

scripts), but now you must activate RCAC on the table. To do so, complete the following steps:

n of the EMPLOYEE table, as shown

## 28 Row and Column Access Control Support in IBM DB2 for i

Example 3-10 Activating RCAC on the EMPLOYEES table /* Active Row Access Control (permissions) */ /* Active Column Access Control (masks) */ the main navigation pane of System i Navigator, click

Run the SQL statements that are shown in Example

3-10 shows the masks that are created in the HR_SCHEMA.

Now that you have created the row permission and the two column masks, RCAC must be activated. The row permission and the two column masks are enabled (last clause in the

<!-- image -->

_Figure 4-69 Index advice with no RCAC_

### Chapter 4. Implementing Row and Colu mn Access Control: Banking example

_Figure 4-68 Visual Explain with RCAC enabled_

3. RCAC enabled. Figure enabled. The index being advised is for the ORDER BY clause. 4-69 shows the index advice for the SQL statement without RCAC

77

2. Figure WHERE clause.

Compare the advised indexes that are provided by the Optimizer without RCAC and with

because the row permission rule becomes part of the

4-68 shows the Visual Explain of the same SQL statement, but with RCAC enabled. It is clear that the implementation of the SQL statement is more complex

<!-- image -->

<!-- image -->

END ENABLE ; RETURN CASE END ENABLE ; RETURN CASE ELSE '*****' END ENABLE ; RETURN CASE ELSE '*****' END ENABLE ; RETURN CASE ELSE '*****' END ENABLE ;

ELSE 'XXX-XX-XXXX' ELSE '*************'

ACTIVATE ROW ACCESS CONTROL

THEN C . CUSTOMER_TAX_ID THEN C . CUSTOMER_TAX_ID

FOR COLUMN CUSTOMER_LOGIN_ID THEN C . CUSTOMER_LOGIN_ID THEN C . CUSTOMER_LOGIN_ID THEN C . CUSTOMER_SECURITY_QUESTION THEN C . CUSTOMER_SECURITY_QUESTION ALTER TABLE BANK_SCHEMA.CUSTOMERS ACTIVATE COLUMN ACCESS CONTROL ;

FOR COLUMN CUSTOMER_SECURITY_QUESTION THEN C . CUSTOMER_SECURITY_QUESTION_ANSWER THEN C . CUSTOMER_SECURITY_QUESTION_ANSWER

FOR COLUMN CUSTOMER_DRIVERS_LICENSE_NUMBER THEN C . CUSTOMER_DRIVERS_LICENSE_NUMBER THEN C . CUSTOMER_DRIVERS_LICENSE_NUMBER THEN C . CUSTOMER_DRIVERS_LICENSE_NUMBER

FOR COLUMN CUSTOMER_SECURITY_QUESTION_ANSWER 124 Row and Column Access Control Support in IBM DB2 for i

WHEN QSYS2 . VERIFY_GROUP_FOR_USER ( SESSION_USER , 'TELLER' ) = 1 WHEN QSYS2 . VERIFY_GROUP_FOR_USER ( SESSION_USER , 'CUSTOMER' ) = 1 WHEN QSYS2 . VERIFY_GROUP_FOR_USER ( SESSION_USER , 'ADMIN' ) = 1 WHEN QSYS2 . VERIFY_GROUP_FOR_USER ( SESSION_USER , 'TELLER' ) = 1 WHEN QSYS2 . VERIFY_GROUP_FOR_USER ( SESSION_USER , 'CUSTOMER' ) = 1 WHEN QSYS2 . VERIFY_GROUP_FOR_USER ( SESSION_USER , 'ADMIN' ) = 1 WHEN QSYS2 . VERIFY_GROUP_FOR_USER ( SESSION_USER , 'CUSTOMER' ) = 1 WHEN QSYS2 . VERIFY_GROUP_FOR_USER ( SESSION_USER , 'ADMIN' ) = 1 WHEN QSYS2 . VERIFY_GROUP_FOR_USER ( SESSION_USER , 'CUSTOMER' ) = 1 WHEN QSYS2 . VERIFY_GROUP_FOR_USER ( SESSION_USER , 'ADMIN' ) = 1 WHEN QSYS2 . VERIFY_GROUP_FOR_USER ( SESSION_USER , 'CUSTOMER' ) = 1

THEN ( 'XXX-XX-' CONCAT QSYS2 . SUBSTR ( C . CUSTOMER_TAX_ID , 8 , 4 ) ) CREATE MASK BANK_SCHEMA.MASK_DRIVERS_LICENSE_ON_CUSTOMERS ON BANK_SCHEMA.CUSTOMERS AS C CREATE MASK BANK_SCHEMA.MASK_LOGIN_ID_ON_CUSTOMERS ON BANK_SCHEMA.CUSTOMERS AS C CREATE MASK BANK_SCHEMA.MASK_SECURITY_QUESTION_ON_CUSTOMERS ON BANK_SCHEMA.CUSTOMERS AS C CREATE MASK BANK_SCHEMA.MASK_SECURITY_QUESTION_ANSWER_ON_CUSTOMERS ON BANK_SCHEMA.CUSTOMERS AS C

<!-- image -->

separation of duties Leverage row permissions on the database Protect columns by defining column masks

For more information: ibm.com /redbooks

REDP-5110-00

IBM Redbooks are developed by the IBM International Technical Support Organization. Experts from IBM, Customers and Partners from around the world create timely technical information based on realistic scenarios. Specific recommendations are provided to help you implement IT solutions more effectively in your environment.

BUILDING TECHNICAL INFORMATION BASED ON PRACTICAL EXPERIENCE

This IBM Redpaper publication provid es information about the IBM i 7.2 Implement roles and feature of IBM DB2 for i Row and Column Access Control (RCAC). It offers a broad description of the function and advantages of controlling access to data in a comprehensive and transparent way. This publication helps you understand the capabilities of RCAC and provides examples of defining, creating, and implementing the row permissions and column masks in a relational database environment.

### INTERNATIONAL TECHNICAL

This paper is intended for database engineers, data-centric application

### SUPPORT

developers, and security officers who want to design and implement

### ORGANIZATION

RCAC as a part of their data control and governance policy. A solid background in IBM i object level security, DB2 for i relational database concepts, and SQL is assumed.

™

## Red paper

## Back cover

®

## Row and Column Access Control Support in IBM DB2 for i

<!-- image -->