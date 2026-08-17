## Contents

| Notices  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  . . vii |
| - |
| Trademarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . viii |
| DB2 for i Center of Excellence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ix |
| Preface  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  . . xi |
| Authors. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . xi |
| Now you can become a published author, too! . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . xiii |
| Comments welcome. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . xiii |
| Stay connected to IBM Redbooks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . xiv |
| Chapter 1. Securing and protecting IBM DB2 data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1 |
| 1.1 Security fundamentals. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 |
| 1.2 Current state of IBM i security. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 |
| 1.3 DB2 for i security controls . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3 |
| 1.3.1 Existing row and column control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4 |
| 1.3.2 New controls: Row and Column Access Control. . . . . . . . . . . . . . . . . . . . . . . . . . . 5 |
| Chapter 2. Roles and separation of duties . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7 |
| 2.1 Roles . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8 |
| 2.1.1 DDM and DRDA application server access: QIBM_DB_DDMDRDA . . . . . . . . . . . 8 |
| 2.1.2 Toolbox application server access: QIBM_DB_ZDA. . . . . . . . . . . . . . . . . . . . . . . . 8 |
| 2.1.3 Database Administrator function: QIBM_DB_SQLADM . . . . . . . . . . . . . . . . . . . . . 9 |
| 2.1.4 Database Information function: QIBM_DB_SYSMON . . . . . . . . . . . . . . . . . . . . . . 9 |
| 2.1.5 Security Administrator function: QIBM_DB_SECADM . . . . . . . . . . . . . . . . . . . . . . 9 |
| 2.1.6 Change Function Usage CL command. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10 |
| 2.1.7 Verifying function usage IDs for RCAC with the FUNCTION_USAGE view . . . . . 10 |
| 2.2 Separation of duties . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10 |
| Chapter 3. Row and Column Access Control . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13 |
| 3.1 Explanation of RCAC and the concept of access control . . . . . . . . . . . . . . . . . . . . . . . 14 |
| 3.1.1 Row permission and column mask definitions . . . . . . . . . . . . . . . . . . . . . . . . . . . 14 3.1.2 Enabling and activating RCAC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16 |
| 3.2 Special registers and built-in global variables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18 |
| 3.2.1 Special registers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18 |
| 3.2.2 Built-in global variables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19 |
| 3.3 VERIFY_GROUP_FOR_USER function. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20 |
| 3.4 Establishing and controlling accessibility by using the RCAC rule text. . . . . . . . . . . . . 21 |
| 3.5 SELECT, INSERT, and UPDATE behavior with RCAC . . . . . . . . . . . . . . . . . . . . . . . . 22 |
| 3.6 Human resources example . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22 |
| 3.6.2 Creating group profiles for the users and their roles. . . . . . . . . . . . . . . . . . . . . . . 23 |
| 3.6.3 Demonstrating data access without RCAC. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24 |
| 3.6.4 Defining and creating row permissions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25 |
| 3.6.5 Defining and creating column masks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26 |
| 3.6.6 Activating RCAC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28 |
| 3.6.8 Demonstrating data access with a view and RCAC . . . . . . . . . . . . . . . . . . . . . . . 32 |

DB2 for i Center of Excellence

Solution Brief IBM Systems Lab Services and Training

<!-- image -->

## Highlights

- -	
		
	
 
		

- -	
		
	
 	


	 

- -		

 ! 	"	#	
- -!	#
	
	
 ""		
		


<!-- image -->

Power Services

## DB2 for i Center of Excellence

Expert help to achieve your business requirements

## We build confident, satisfied clients

No one else has the vast consulting experiences, skills sharing and renown service offerings to do what we can do for you.

Because no one else is IBM.

With combined experiences and direct access to development groups, we're the experts in IBM DB2® for i. The DB2 for i Center of Excellence (CoE) can help you achieve—perhaps reexamine and exceed—your business requirements and gain more confidence and satisfaction in IBM product data management products and solutions.

## Who we are, some of what we do

Global CoE engagements cover topics including:

- r Database performance and scalability
- r Advanced SQL knowledge and skills transfer
- r Business intelligence and analytics
- r DB2 Web Query
- r Query/400 modernization for better reporting and analysis capabilities
- r Database modernization and re-engineering
- r Data-centric architecture and design
- r Extremely large database and overcoming limits to growth
- r ISV education and enablement

## Preface

This IBM® Redpaper™ publication provides information about the IBM i 7.2 feature of IBM DB2® for i Row and Column Access Control (RCAC). It offers a broad description of the function and advantages of controlling access to data in a comprehensive and transparent way. This publication helps you understand the capabilities of RCAC and provides examples of defining, creating, and implementing the row permissions and column masks in a relational database environment.

This paper is intended for database engineers, data-centric application developers, and security officers who want to design and implement RCAC as a part of their data control and governance policy. A solid background in IBM i object level security, DB2 for i relational database concepts, and SQL is assumed.

This paper was produced by the IBM DB2 for i Center of Excellence team in partnership with the International Technical Support Organization (ITSO), Rochester, Minnesota US.

<!-- image -->

<!-- image -->

Jim Bainbridge is a senior DB2 consultant on the DB2 for i Center of Excellence team in the IBM Lab Services and Training organization. His primary role is training and implementation services for IBM DB2 Web Query for i and business analytics. Jim began his career with IBM 30 years ago in the IBM Rochester Development Lab, where he developed cooperative processing products that paired IBM PCs with IBM S/36 and AS/.400 systems. In the years since, Jim has held numerous technical roles, including independent software vendors technical support on a broad range of IBM technologies and products, and supporting customers in the IBM Executive Briefing Center and IBM Project Office.

Hernando Bedoya is a Senior IT Specialist at STG Lab Services and Training in Rochester, Minnesota. He writes extensively and teaches IBM classes worldwide in all areas of DB2 for i. Before joining STG Lab Services, he worked in the ITSO for nine years writing multiple IBM Redbooks® publications. He also worked for IBM Colombia as an IBM AS/400® IT Specialist doing presales support for the Andean countries. He has 28 years of experience in the computing field and has taught database classes in Colombian universities. He holds a Master's degree in Computer Science from EAFIT, Colombia. His areas of expertise are database technology, performance, and data warehousing. Hernando can be contacted at hbedoya@us.ibm.com .

## Authors

<!-- image -->

Chapter 1.

## 1

## Securing and protecting IBM DB2 data

Recent news headlines are filled with reports of data breaches and cyber-attacks impacting global businesses of all sizes. The Identity Theft Resource Center 1 reports that almost 5000 data breaches have occurred since 2005, exposing over 600 million records of data. The financial cost of these data breaches is skyrocketing. Studies from the Ponemon Institute 2 revealed that the average cost of a data breach increased in 2013 by 15% globally and resulted in a brand equity loss of $9.4 million per attack. The average cost that is incurred for each lost record containing sensitive information increased more than 9% to $145 per record.

Businesses must make a serious effort to secure their data and recognize that securing information assets is a cost of doing business. In many parts of the world and in many industries, securing the data is required by law and subject to audits. Data security is no longer an option; it is a requirement.

This chapter describes how you can secure and protect data in DB2 for i. The following topics are covered in this chapter:

- -Security fundamentals
- -Current state of IBM i security
- -DB2 for i security controls

1 http://www.idtheftcenter.org

2 http://www.ponemon.org/