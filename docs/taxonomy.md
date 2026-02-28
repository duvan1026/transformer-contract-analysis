# Taxonomía Operativa de Cláusulas

## **1. Ley aplicable (Governing Law)**

### **Definición:**
Cláusulas contractuales que establecen explícitamente la ley estatal, nacional o provincial que regirá la interpretación, validez, ejecución o efectos legales del contrato.

Estas cláusulas determinan el marco jurídico bajo el cual se interpretará el contrato en caso de disputa o análisis legal.

### **Criterio Semántico Central**
El fragmento debe contener:

* Una referencia explícita a una jurisdicción legal (por ejemplo, un estado, país o provincia), y

*  Una expresión lingüística que indique que dicha ley rige, gobierna, interpreta o regula el contrato.

En el dataset, esta intención aparece mediante formulaciones reales como:

Ejemplos reales extraídos del dataset:

* “This Agreement is to be construed according to the laws of the State of Illinois.”
* “This Agreement is governed by English law.”
* “This Agreement shall be governed by the laws of the State of Texas.”
* “This Amendment shall be governed by and construed in accordance with the laws of Japan.”
* “This Agreement shall be governed and construed according to the laws of the State of Kansas.”
* “Any disputes … shall be governed by, and construed in accordance with, the Laws of the State of Delaware.”
* “This Agreement is governed by the laws of Victoria, Australia.”

### **Incluye:**
Se clasifican como **Governing Law** los fragmentos que designen explícitamente la ley que rige el contrato o su interpretación.

Esto incluye fragmentos que contengan expresiones como:

**Patrones lingüísticos reales identificados en el dataset:**

* “governed by the laws of …”
* “governed by … law”
* “shall be governed by …”
* “governed by and construed in accordance with …”
* “construed in accordance with the laws of …”
* “construed according to the laws of …”
* “interpreted in accordance with the laws of …”
* “interpreted according to the laws of …”
* “governed and construed according to the laws of …”
* “validity, construction, interpretation … shall be governed by the laws of …”

También se incluyen los casos en los que la cláusula menciona adicionalmente:

* interpretación (interpretation)
* construcción (construction)
* ejecución (performance)
* validez (validity)
* aplicación (enforcement)

siempre que se indique explícitamente la jurisdicción aplicable.

**Nota interpretativa:** Como regla práctica, si el fragmento permite identificar claramente la jurisdicción bajo la cual se rige el contrato, se clasifica como Governing Law.

---

## **2. Terminación por conveniencia (Termination for Convenience)**

### **Definición:**

Cláusulas contractuales que permiten a una de las partes dar por terminado el contrato sin necesidad de que exista incumplimiento previo o causa específica.

Generalmente, este tipo de terminación puede ejercerse mediante notificación escrita y, en algunos casos, respetando un período mínimo o un plazo de preaviso.

### **Criterio Semántico Central**

El fragmento debe indicar explícitamente que una de las partes puede terminar el contrato sin necesidad de incumplimiento o causa específica.

En el dataset, esta intención aparece mediante expresiones como:

* “terminate … for convenience”
* “terminate … without cause”
* “terminate … for its convenience”
* “terminate … at its sole discretion”
* “terminate … for any reason or no reason”

Estas formulaciones indican que la terminación depende únicamente de la decisión de una de las partes.

### **Incluye:**

Se clasifican como Termination for Convenience los fragmentos que indiquen explícitamente que una de las partes puede terminar el contrato sin necesidad de incumplimiento o causa específica.

Esto incluye fragmentos que contengan expresiones como:

* “terminate … for convenience”
* “terminate … for its convenience”
* “terminate … without cause”
* “terminate … with or without cause”
* “terminate … for any reason or no reason”
* “terminate … at its sole discretion”

También se incluyen los casos en los que la terminación se puede ejercer mediante notificación escrita y la decisión depende únicamente de una de las partes.

---

## **3. Derechos de auditoria (Audit Rigth)**

### **Definición:**

Cláusulas contractuales que otorgan a una de las partes el derecho a auditar, inspeccionar o revisar los libros, registros o información de la otra parte con el fin de verificar el cumplimiento de las obligaciones establecidas en el contrato, incluyendo pagos, actividades o desempeño contractual.

### **Criterio Semántico Central**

El fragmento debe indicar explícitamente que una de las partes puede auditar, inspeccionar o acceder a registros o libros de la otra parte.

En el dataset, esta intención aparece mediante formulaciones como:

Ejemplos extraídos del dataset:

* “Google may audit Distributor's relevant records to confirm Distributor's compliance with this Agreement.”
* “to inspect and audit the applicable records and books maintained by Valeant relating to the Valeant Activities”
* “to inspect and audit the applicable records and books maintained by Dova for purposes of verifying Dova's payment obligations”
* “Dova's audit rights shall include interviewing Sales Representatives and other employees of Valeant.”
* “make available for inspection the records”

Estas formulaciones indican que una parte tiene autorización formal para revisar información de la otra parte.

### **Incluye:**

Se clasifican como Audit Rights los fragmentos que otorguen explícitamente el derecho de auditar o inspeccionar registros de la otra parte.

Patrones lingüísticos reales identificados en el dataset:

* “may audit … records”
* “right … to inspect and audit … records and books”
* “inspect and audit the applicable records and books”
* “audit rights shall include …”
* “make available for inspection the records”
* “inspect and audit … books of account”

También se incluyen fragmentos donde el derecho de auditoría tenga como propósito explícito:

* verificar cumplimiento del contrato
Ejemplo:
“may audit Distributor's relevant records to confirm Distributor's compliance with this Agreement”

* verificar obligaciones de pago
Ejemplo:
“to inspect and audit the applicable records and books maintained by Dova for purposes of verifying Dova's payment obligations”

* confirmar la exactitud o integridad de registros
Ejemplo:
“to confirm the accuracy and completeness of such records and books of account”

---

## **4. No competencia (Non-compete)**

### **Definición:**

Cláusulas contractuales que restringen la capacidad de una de las partes para participar en actividades, negocios o relaciones comerciales que compitan directa o indirectamente con la otra parte o con el objeto del contrato.

Estas restricciones pueden aplicarse durante la vigencia del contrato y, en muchos casos, durante un período posterior a su terminación.

### **Criterio Semántico Central**

El fragmento debe indicar explícitamente que una de las partes no puede competir, participar en negocios competidores, desarrollar productos competidores o colaborar con competidores.

En el dataset, esta intención aparece mediante formulaciones reales como:

Ejemplos reales extraídos de los spans:

* “will not … engage in any enterprise or business … which manufactures … develops, markets, or sells products falling within the Field of Agreement” 

* “ISA agrees not to … engage or have a financial interest in any Rubicon Competing Business” 

* “Distributor will not market … a computer program which competes in any way with the Products” 

* “The Developer shall not develop, maintain or market a similar platform and will not compete with the Client” 

* “Endorser's duty not to compete with the business of MusclePharm shall continue for a period of one year” 

* “no Participant will engage in any business … that might be competitive with the business of the Joint Venture” 


Estas formulaciones indican una prohibición directa de competir o participar en actividades competidoras.

### **Incluye:**

Se clasifican como Non-Compete los fragmentos que establezcan restricciones explícitas como:

#### **1. Prohibición de participar en negocios competidores**
    
Ejemplos:

* “will not … engage in any enterprise or business … which manufactures … develops, markets, or sells products falling within the Field of Agreement” 

* “not … engage or have a financial interest in any … Competing Business”

#### **2. Prohibición de desarrollar, vender o comercializar productos competidores**

Ejemplos:

* “Distributor will not market … a computer program which competes in any way with the Products” 

* “Surgical shall not promote, market or sell any products … which directly or indirectly compete with the Product” 

#### **3. Prohibición de desarrollar o mantener productos o tecnologías competidoras**

Esto ocurre cuando una parte no puede crear o mantener soluciones similares.

Ejemplos:

* “The Developer shall not develop, maintain or market a similar platform and will not compete with the Client”

#### **4. Prohibición de tener interés financiero o participación en competidores**

Incluye restricciones sobre inversión, propiedad o participación económica.

Ejemplos:

* “ISA agrees not to … have a financial interest in any Rubicon Competing Business”

---

## **5. Limitación de responsabilidad (Limitation of Liability)**

**Nota**: Esta categoría unifica las categorías originales de CUAD **Cap on Liability** y **Uncapped Liability**, ya que ambas definen el alcance de la responsabilidad legal de las partes, ya sea estableciendo un límite máximo o indicando excepciones donde la responsabilidad no está limitada.

### **Definición:**

Cláusulas contractuales que definen el alcance cuantitativo de la responsabilidad económica de las partes, estableciendo límites máximos de indemnización (caps) o declarando explícitamente las situaciones, conductas o materias en las que dicha responsabilidad no tiene límite financiero (uncapped).

### **Criterio Semántico Central**

El fragmento debe indicar una restricción monetaria mediante sumas fijas, porcentajes o fórmulas, o bien establecer una excepción donde la limitación de responsabilidad no aplica.

En el dataset, esta intención aparece mediante formulaciones como:

Ejemplos extraídos del dataset:

* "each party's total liability... is limited to the greater of: (a) [ * ] Euros... and (b) [ * ]% of the total payment due"
* "Nothing in this Agreement shall exclude or limit either party's liability for... fraud or fraudulent misrepresentation"
* "total liquidated damages as per this Section shall not exceed 5% of the Product Price."
* "this Subsection will not limit the liability of a Party for... infringement by the Party on the intellectual property"
* "liability for damages... shall be limited to one (1) month's fees and the remaining portion of any prepaid fees."

Estas formulaciones indican o bien un techo financiero o la preservación de la responsabilidad plena ante eventos específicos.

### **Incluye:**

Se clasifican bajo esta categoría los fragmentos que definen el alcance monetario de la responsabilidad, ya sea estableciendo un techo máximo o señalando las excepciones donde dicho techo desaparece.

#### **1. Limitaciones Cuantitativas (Caps):**

Fragmentos que restringen el resarcimiento mediante sumas fijas, variables o porcentuales. Patrones reales identificados:

* “total liability... is limited to the greater of...”
* “shall not exceed the lesser of $50,000... or the amount paid”
* “liability... shall be limited to [**] US Dollars”
* “maximum liability... shall be limited to the amount of fees paid”
* “not exceed 5% of the Product Price”

#### **2. Excepciones a la Limitación (Uncapped Liability):**

Fragmentos que declaran que el límite no es aplicable o que la responsabilidad es plena debido a la naturaleza de la conducta o la materia. Patrones reales identificados:

* “Nothing in this Agreement shall exclude or limit... liability for...”
* “this Subsection will not limit the liability of a Party for...”
* “provided however that such limitation shall not apply to the extent...”
* “EXCEPT IN THE EVENT OF... WILLFUL MISCONDUCT OR FRAUD...”

#### **3. Supuestos específicos de responsabilidad ilimitada:**

Se incluyen aquellos fragmentos donde se especifica que la limitación no protege los siguientes activos críticos:

* Integridad física: “death or personal injury resulting from the negligence”
* Propiedad Intelectual: “infringement by the Party on the intellectual property”
* Indemnidades: “liability under Clause 10 (Indemnities)”
* Confidencialidad: “unauthorized disclosure or use... of Confidential Information”
