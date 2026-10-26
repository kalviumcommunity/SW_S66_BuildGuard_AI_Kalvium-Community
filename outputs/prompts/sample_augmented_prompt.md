# Augmented Prompt & Context Assembly Report

**Generated:** `2026-10-26T13:44:44Z`  
**Query:** *What is the minimum fire resistance rating required for walls separating a boiler room from other parts of a building?*

## 1. Token Budget Allocation

| Budget Metric | Tokens | % of Max Window | Description |
|:---|---:|---:|:---|
| **Max Window Budget** | **4,096** | 100.0% | Total model context window limit |
| **Reserved for Generation** | **512** | 12.5% | Guaranteed headroom for model output |
| System Instructions | 189 | 4.6% | Grounding constraints and persona rules |
| Query + Framing Overhead | 120 | 2.9% | User question and instruction templates |
| **Available Context Budget** | **3,275** | 80.0% | Maximum allowed for injected chunks |
| **Used Context Tokens** | **3,192** | 77.9% | Actual tokens consumed by chunks |
| Total Prompt Tokens | 3,506 | 85.6% | Input tokens sent to API (system + user) |
| Remaining Safety Headroom | 78 | 1.9% | Unused token capacity |
| **Overall Budget Utilization** | **—** | **98.1%** | (Input prompt + reserved answer) / Max |

- **Chunks Considered:** 6
- **Chunks Injected:** 6
- **Chunks Truncated:** 0
- **Chunks Skipped:** 0

## 2. Injected Sources & Markers

| Marker | Chunk ID | Source Doc | Vol | Page | Clause | Section | Sim Score | Re-rank Score | Tokens | Status |
|:---|:---|:---|:---:|:---:|:---|:---|---:|---:|---:|:---:|
| `[1]` | `nbc-v1-p314-001012` | nationalbuilding01.pdf | 1 | 314-315 | 6.1.1.2 | Part 4 — Fire and Life Safety | 0.8264 | — | 447 | Full |
| `[2]` | `nbc-v1-p279-000915` | nationalbuilding01.pdf | 1 | 279-280 | 3.4.7.3 | Part 4 — Fire and Life Safety | 0.7682 | — | 547 | Full |
| `[3]` | `nbc-v1-p272-000892` | nationalbuilding01.pdf | 1 | 272-273 | — | Part 4 — Fire and Life Safety | 0.7497 | — | 544 | Full |
| `[4]` | `nbc-v1-p275-000900` | nationalbuilding01.pdf | 1 | 275 | 3.4.4 | Part 4 — Fire and Life Safety | 0.7781 | — | 554 | Full |
| `[5]` | `nbc-v1-p265-000868` | nationalbuilding01.pdf | 1 | 265-266 | 2.25 | Part 4 — Fire and Life Safety | 0.8084 | — | 553 | Full |
| `[6]` | `nbc-v1-p274-000896` | nationalbuilding01.pdf | 1 | 274 | — | Part 4 — Fire and Life Safety | 0.7990 | — | 547 | Full |

## 3. Grounding Instructions (System Message)

```text
You are an expert staff-support assistant answering technical compliance questions about the National Building Code of India (NBC).
Follow these strict operational rules:
1. Answer the question using ONLY the factual evidence directly stated in the RETRIEVED BUILDING CODE CONTEXT below.
2. Do NOT extrapolate, speculate, assume, or introduce outside knowledge not corroborated by the provided context.
3. Attribute every factual claim, requirement, rating, dimension, or specification using the bracketed source markers (e.g. [1], [2]) or explicit citations such as [Volume 1, PDF p. 314-315].
4. If the provided context does NOT contain sufficient factual evidence to answer the question reliably, reply with this exact fallback message:
   "I don't have enough information to answer that reliably based on the provided National Building Code context."
5. Keep the response professional, precise, and concise (typically 2 to 4 sentences).
```

## 4. Rendered User Prompt with Injected Context

```text
=== RETRIEVED BUILDING CODE CONTEXT ===
[1] | Doc: nationalbuilding01.pdf | Volume: 1 | Page: 314-315 | Clause: 6.1.1.2 | Section: Part 4 — Fire and Life Safety
Score: Similarity: 0.8264
------------------------------------------------------------
120 min fire resistance rating wall. Such areas . shall be provided with fire doors, to be kept b) All exterior openings in a boiler room or closed and shall be posted with a sign on each rooms containing central heating equipment, PART 4 FIRE AND LIFE SAFETY 59 if located below an opening in another storey compartment and on use of adjacent or if less than 3 m from other doors or windows compartments as temporary means of refuge. of the same building, shall be protected by a All compartments shall be divided with self- fire assembly as in 3.4.5. Such assemblies shall closing (door closers) fire doors with be of fixed, automatic or self-closing type. electromagnetic hold open. A coordinator segs ey as shall be provided to sequence the closing of 6.3 Institutional Buildings (Group C) Pro. a 8 double leaf in case of emergency. ire Prevention d) Doors in fire resistant walls shall be so No combustible material of any kind shall be stored or installed that these may normally be kept in used in any building or section thereof used for open position, but will close automatically. institutional occupancy, except as necessary to normal Corridor door openings shall be not less than occupancy and useofthe building. 2.0 m in width of double swing double leaf . , type door. A coordinator shall be provided as 6.3.2 Life Safety . . ife Safety above, for closing of double leaf in case of a) The common path oftravel shall be 30 m. The emergency. maximum dead end of corridor distance shall e) Exits and other features for penal and mental not exceed 6 m. eee nr institutions, and custodial institutions shall be b) Principle of progressive horizontal evacuation the same as specified for hospitals [see is of paramount consideration for hospital 6.3.2 (g)], in so far as applicable.
------------------------------------------------------------

[2] | Doc: nationalbuilding01.pdf | Volume: 1 | Page: 279-280 | Clause: 3.4.7.3 | Section: Part 4 — Fire and Life Safety
Score: Similarity: 0.7682
------------------------------------------------------------
3.4.8.1 Air conditioning and mechanical ventilation —_3.4.8.2.4 The air handling unit room shall not be used requirements of different rooms or areas in any for storage of any combustible materials. occupancy shall be as given in Part 8 ‘Building Services, Section 1 Lighting and Natural Ventilation’ 3.4.8.3 Duct work and ‘Section 3 Air conditioning, Heating and 3.4.8.3.1 Air ducts serving main floor areas, corridors, Mechanical Ventilation’ of the Code. etc, shall not pass through the exits/exit passageway/ Air conditioning and ventilating systems shall be so exit enclosure. Exits and lift lobbies, etc, shall not be installed and maintained as to minimise the danger of used as return air passage. spread of fire, smoke or fumes from one floor to other —_3.4.8.3.2 As far as possible, metallic ducts shall be used or from outside to any occupied building or structure. even for the return air instead of space above the false Many high-rise buildings integrate smoke management ceiling. systems into their conventional HVAC systems. Insuch —_3.4.8.3.3 Wherever the ducts pass through fire walls or installation, it requires special design considerations, floors, the opening around the ducts shall be sealed 24 NATIONAL BUILDING CODE OF INDIA 2016 with materials having fire resistance rating of the 3.4.9.2.2 Further, the following additional aspects may compartment. Such duct shall also be provided with be taken into account in the location of boiler room: fire dampers at all fire walls and floors unless such a) The boilers shall be installed in a fire resisting ucts are eauired perform for iesafety operations room of 180 min fire resistance rating. and in such case fire damper maybe avoided at fire . . . wall and floor while integrity of the duct shall be ) vance dove of 120be provided with 8 maintained with 120 min fire resistance rating to allow rating. the emergency operations for fire safety requirements. . La c) The boiler room shall be provided with its 3.4.8.3.4 The ducting within compartment would dedicated natural or mechanical ventilation require minimum fire resistance rating of30 min. Such system.
------------------------------------------------------------

[3] | Doc: nationalbuilding01.pdf | Volume: 1 | Page: 272-273 | Section: Part 4 — Fire and Life Safety
Score: Similarity: 0.7497
------------------------------------------------------------
3.2.4.1 When any building is so situated that it extends . to more than one fire zone, it shall be deemed to be in 3.1.14 A certificate of occupancy shall be necessary, _the fire zone in which the major portion of the building as required under Part 2 ‘Administration’ of the Code, or structure is situated. before any change is effected in the character of occupane. *tan Soitatin 3.2.4.2 When any building is so situated that it extends Pancy y e. equally to more than one fire zone, it shall be deemed PART 4 FIRE AND LIFE SAFETY 17 to be in the fire zone having more hazardous occupancy The fire resistance of a building or its structural and buildings. non-structural elements is expressed in minutes against ., . ype viene a specified fire load which is expressed in keal/m?, and 3.2.5 Temporary Buildings or Structures against a certain intensity of fire. The fire-resistance 3.2.5.1 Temporary buildings and structures shall be test for structural element shall be done in accordance permitted only in Fire Zones No. | and 2 as the case with accepted standards [4(2)]. For the purpose ofthe may be, according to the purpose for which these are Code, the types of construction according to fire to be used, by special permit from the Authority for a resistance shall be classified into four categories, limited period and subject to such conditions as may namely, Type 1 Construction, Type 2 Construction, be imposed in the permit. Type 3 Construction and Type 4 Construction. The 3.2.5.2 Such buildings and temporary structures shall minimum fire resistance ratings of structural and none : , structural members for various types of construction be completely removed on the expiry of the period : A . : . shall be as given in Table 1. specified in the permit. 3.2.5.3 Adequate fire precautionary measures in the Non-combustible. materials should be used for : construction of buildings, and the internal walls of construction oftemporary structures and Pandals shall . . . . . staircase enclosures should be of brick work or be taken in accordance with good practice [4(4)]. reinforced concrete or any other material of 3.2.6 Restrictions on the Type ofConstructionfor New construction with minimum of 120 min rating.
------------------------------------------------------------

[4] | Doc: nationalbuilding01.pdf | Volume: 1 | Page: 275 | Clause: 3.4.4 | Section: Part 4 — Fire and Life Safety
Score: Similarity: 0.7781
------------------------------------------------------------
3.4.4.2 Floor area ratio NOTES = : Fi 1 The FAR values given in this table are subject to overall The comparative floor area ratios for different restrictions on the heights of buildings in the case ofeducational, occupancies and types of construction are given in institutional, assembly, storage and hazardous occupancies as Table 2 (see also Part 3 ‘Development Control Rules specified in col 2 of Table 7. and General Building Requirements’ of the Code). 2 This table has been prepared, taking into account the : combustible content in the different occupancies as well as the 3.4.4.3 The requirements of open space around the fire resistance offered by the type of construction. building(s) shall be in accordance with Part 3 3 This table should be modified by the Authority, taking into ‘Development Control Rules and General Building account the other aspects as given below: Requirements’ of the Code. a) Density in terms of dwelling units per hectare; 3.4.5 Openings in Fire Resistant Walls and Floors b) Traffic|considerations; : oo : . . c) Parking spaces; 3.4.5.1 At the time of designing openings in fire ) hess Pn : : d) Local fire-fighting facilities; and resistant walls and floors, particular attention shall be on . * Pamcty e) Water supply, drainage and sanitation requirements. paid to all such factors which limit fire spread through 4 The FAR values specified in this table may be increased by these openings and maintain fire rating of the structural 20 percent for the following services: members. a) A basement or cellar space under a building constructed 3.4.5.2 For Types | to 3 constructions, a doorway or on stilts and used as a parking space and air conditioning a . plant room used as accessory to the principal use; opening in a fire resistant wall on any floor shall be 5 . nan a ith . height/width b) Watchman’s booth, pump house, garbage shaft, electric limited to 5.6 m? in area wi ha maximum height/wid cabin or substation and other utility structures meant of 2.75 m. Every wall opening shall be protected with for the services of the building under considerations; fire-resisting doors, having the fire rating of not less c) Projections and accessory buildings as specifically than 120 min.
------------------------------------------------------------

[5] | Doc: nationalbuilding01.pdf | Volume: 1 | Page: 265-266 | Clause: 2.25 | Section: Part 4 — Fire and Life Safety
Score: Similarity: 0.8084
------------------------------------------------------------
Se . . oe . resistance rated wall having fire protected openings, 2 The fire resistance rating shall be specified in terms of minutes. . . . . . . which restricts the spread of fire and extends 3 Fire resistance rating for non-structural material/assembly . shall bear a label ofcompliance to such rating as per the approval Continuously from the foundation to the roof (and of competent authority based on testing and evaluation. The through the roof at least 1m above the roof in case label shall be permanently affixed to the material/assembly and gf combustible roof), with sufficient structural may carry other relevant details such as name and type of the stability under fire conditions to allow collapse of product, and manufacturer’s details. . . . . t construction on one side or either side without 2.30 Fire Resistant Wall— Fire resistance rated wall, collapse of the wall. having opening(s) with specified fire resistant rating, a. which restricts the spread of fire from one part of a _2-35 Floor Area (Gross) — The area of the floor within building to another part of the same building. the inside perimeter of the outside walls of the floor of . . . . the building under consideration with no deductions 2.31 Fire Separation — The distance in metre, for corridors and passage-ways, stairs, closets, thickness measured from the external wall of the building of interior walls, columns, lifts and building shafts or concerned to the external wall ofany other building on other features. the site, or from other site, or from the opposite side of street or other public space for the purpose of 2.36 Floor Area Ratio (FAR) — The quotient obtained preventing the spread of fire. by dividing the total covered area (plinth area) on all 10 NATIONAL BUILDING CODE OF INDIA 2016 floors by the area of the plot: 2.43.6 Non-transit Occupancy— Occupancy not under the control of the system operating authority. _ Total covered area of all floors FAR =area 2.43.7 Point ofSafety— One ofthe following: (a) An . . . enclosed exit that leads to a public way or safe location 2.37 Fire Exit Hardware — A door-latching assembly outside the station, trainway, or vehicle, (b) An at- incorporating an actuating member or panic bar that A . : .
------------------------------------------------------------

[6] | Doc: nationalbuilding01.pdf | Volume: 1 | Page: 274 | Section: Part 4 — Fire and Life Safety
Score: Similarity: 0.7990
------------------------------------------------------------
Structural Element Fire Resistance Ratings (min) for Type of Construction ——o> Type 1 Type 2 Type 3 Type 4 io) (2) G3) (4) (5) (6) i) Exterior walls: a) Fire separation less than 3.7 m: 1) Bearing 240 120 120 60 2) Non-bearing 120 90 60 60 b) Fire separation of 3.7 m or more but less than 9 m: 1) Bearing 240 120 120 60 2) Non-bearing 90 60 60 60 c) Fire separation of9m or more: 1) Bearing 240 120 120 60 2) Non-bearing 60 60 60 60 ii) Fire separation assemblies (like fire check doors) 120 120 120 120 iii) Fire enclosures of exits 120 120 120 120 iv) Shafts for services, lift hoistway and refuse chutes 120 120 120 120 y) Vertical separation between adjacent tenant spaces 60 60 60 60 vi) Dwelling unit separation: a) Load bearing 120 120 60 60 b) Non-load bearing 60 60 30 30 vii) Interior bearing walls, bearing partitions, columns, beams, girders, trusses (other than roof trusses) and framing: a) Supporting more than one floor 240 120 120 120 b) Supporting one floor only 180 90 60 60 c) Supporting a roof only 180 90 60 60 viii) Walls supporting structural members 180 90 60 60 ix) Floor construction 120 90 60 60 x) Roofconstruction: a) 5 mor less in height to lowest member 120 90 60 60 b) More than 5 m but less than 6.7 m in height to lowest member 60 60 60 60 c) 6.7 mor more in height to lowest member 0 0 0 0 NOTES 1 The above fire resistance rating shall be required to achieve the respective type of construction unless otherwise specified in the respective clauses for different applications/use. 2 In caseoflift bank, the partition wall, if any, need not be of fire rating specified in this
------------------------------------------------------------
=======================================

GROUNDING INSTRUCTIONS:
- Answer using ONLY the retrieved context above.
- Cite your sources for every factual assertion using markers like [1], [2] or [Volume X, PDF p. Y].
- If the context does not provide sufficient evidence, output: "I don't have enough information to answer that reliably based on the provided National Building Code context.".

USER QUESTION:
What is the minimum fire resistance rating required for walls separating a boiler room from other parts of a building?

COMPLIANCE ANSWER:
```