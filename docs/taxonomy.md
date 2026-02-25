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



