# Rank Homes

Rank Homes ranks rental listings by visual similarity to reference images of kitchens and bathrooms.

## Language

**Rental listing**:
A rental offer for a home, with its own collection of photos and, when available, a source link.

**Listing photo**:
A photo belonging to a rental listing. Several photos can show the same room or object.

**Reference image**:
An image supplied as a visual example of the desired appearance of target objects in a kitchen or bathroom.

**Reference library**:
A collection of reference images approved by the user to represent the desired appearance of target objects in kitchens and bathrooms.

**Reference alternative**:
An acceptable appearance of a target object represented in the reference library. A rental listing may match alternatives from different reference apartments for different object classes and rooms.

**Matching photo**:
A listing photo shown with a search result as evidence of its visual similarity to a reference image.

**Room object detection**:
Identifying target objects in a room photo with a class label and a bounding box for each instance.

**Target object**:
An object, fixture, or room surface selected for visual comparison in a kitchen or bathroom, such as a sink, shower door, countertop, floor, or wall.

**Shower stall**:
A bounded showering area, including the enclosure visible around it. Its door can also be a separate target object.

**Shower door**:
A movable panel that provides access to a showering area.

**Object crop**:
A rectangular region of a reference image or listing photo containing a detected target object. It can also contain background or parts of other objects within that rectangle.

**Listing similarity score**:
An aggregate measure of a rental listing's visual similarity to the reference library, based on comparisons between object crops of the same class and room. It is not a probability of user interest or a measure of the home's condition.

**Room similarity score**:
A measure of a rental listing's visual similarity to the reference library for one room type, combining the object classes represented in that room's references. Missing comparable objects mean missing visual evidence, not proof that the objects are physically absent.

**Object class similarity score**:
A measure of the strongest visual match between a rental listing's object crops and reference alternatives of the same class and room. Each expected class contributes once to its room's similarity score, regardless of the number of photos or detected instances.

**Listing similarity threshold**:
A configurable cutoff applied to the listing similarity score to select rental listings considered sufficiently similar to the reference library.
